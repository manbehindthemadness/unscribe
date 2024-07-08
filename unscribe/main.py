"""
Hold our general logic.
"""
import os
import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from mcraft import TNet
from omegaconf import OmegaConf
from unscribe.saicinpainting.evaluation.utils import move_to_device
from torch.utils.data._utils.collate import default_collate  # noqa
from unscribe.saicinpainting.training.trainers import load_checkpoint
from unscribe.saicinpainting.evaluation.data import scale_image, pad_img_to_modulo
from .refiner import refine_predict
try:
    from utils import (
        create_opacity_mask, visualize_polys, fetch_contours, mask_from_polys, convert_to_image,
    )
except ImportError:
    from .utils import (
        create_opacity_mask, visualize_polys, fetch_contours, mask_from_polys, convert_to_image,
    )

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

models_path = Path('~/.cache/dscribe/').expanduser()

MODES = ['scramble', 'remove']


class Remover(TNet):
    """
    Removes text and watermarks.
    """
    def __init__(
            self,
            gpu: str = 'cpu',
            poly: bool = False,
            refine: bool = False,
            debug: bool = False,
            show_mats: bool = False,
            lama_refine: bool = False
    ):
        cuda = True
        if gpu == 'cpu':
            cuda = False
        super().__init__(
            cuda=cuda,
            poly=poly,
            refine=refine
        )
        self.gpu = gpu
        self.lama_refine = lama_refine
        self.debug = debug
        self.show_mats = show_mats
        self.device = torch.device("cuda:0" if cuda else "cpu")
        self.d_print('DEVICE', self.device)
        self.models_path = models_path
        self.config_path = self.models_path / 'config.yaml'
        with open(self.config_path.as_posix(), 'r') as f:
            self.config = OmegaConf.create(yaml.safe_load(f))
            f.close()
        self.config.training_model.predict_only = True
        self.config.visualizer.kind = 'noop'
        self.checkpoint_path = self.models_path / 'describe_lama.ckpt'
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f'checkpoint not found at: {self.checkpoint_path.as_posix()}')
        self.model = load_checkpoint(
            self.config,
            self.checkpoint_path.as_posix(),
            strict=False,
            map_location=self.device
        ).to(self.device)
        self.model.freeze()

    def d_print(self, *args, **kwargs):
        """
        Just a basic message sender.
        """
        if self.debug:
            print(*args, **kwargs)
        return self

    @staticmethod
    def process_mat(mat: np.ndarray) -> np.ndarray:
        """
        THis is an adaptation of from unscribe.saicinpainting.evaluation.data.load_image
        """
        if mat.ndim == 3:
            mat = np.transpose(mat, (2, 0, 1))
        if mat.ndim == 2:
            mat = np.expand_dims(mat, axis=0)
        mat = mat.astype('float32') / 255
        return mat

    def load_mat(
            self,
            mat: np.ndarray = None,
            scale_factor: [int, float] = None,
            pad_out_to_modulo: [int, float] = 8,
            low_clamp: float = 0.1,
            high_clamp: float = 1.0,
            passes: int = 15,
            lr: float = 0.002,  # learning rate
            min_side: int = 512,  # all sides of image on all scales should be >= min_side / sqrt(2)
            max_scales: int = 3,  # max number of downscaling scales for the image-mask pyramid
            px_budget: int = 1800000,  # pixels budget. Any image will be resized to satisfy height*width <= px_budget
            mode: MODES = 'scramble',
    ) -> np.ndarray:
        """
        Load and process an image mat.
        """
        self.d_print('input image size', mat.shape)
        rgb_mat = mat
        _, polys, mask_mat = self.forward(mat)  # bboxes, polys, score_text.
        mask_mat = create_opacity_mask(mask_mat, half=True, low_clamp=low_clamp, high_clamp=high_clamp)
        new_height, new_width = rgb_mat.shape[:2]
        mask_mat = cv2.resize(mask_mat, (new_width, new_height))
        match mode:
            case 'scramble':
                mask_mat = fetch_contours(rgb_mat, mask_mat, polys)
            case 'remove':
                mask_mat = mask_from_polys(mask_mat, polys)

        if self.show_mats:
            cv2.imshow('mask', mask_mat)
            original = np.array(rgb_mat)
        processed_rgb_mat, processed_mask_mat = [
            self.process_mat(rgb_mat),
            self.process_mat(mask_mat)
        ]
        if scale_factor:
            processed_rgb_mat, processed_mask_mat = [
                scale_image(processed_rgb_mat, scale_factor),
                scale_image(processed_mask_mat, scale_factor, interpolation=cv2.INTER_NEAREST)
            ]
        un_pad_size = None
        if pad_out_to_modulo:
            un_pad_size = processed_rgb_mat.shape[1:]
            processed_rgb_mat, processed_mask_mat = [
                pad_img_to_modulo(processed_rgb_mat, pad_out_to_modulo),
                pad_img_to_modulo(processed_mask_mat, pad_out_to_modulo)
            ]
        dataset = {
            'image': processed_rgb_mat,
            'mask': processed_mask_mat,
        }

        if self.lama_refine:
            dataset['unpad_to_size'] = un_pad_size
            batch = default_collate([dataset])
            assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
            inpainted_mat = refine_predict(
                batch,
                self.model,
                n_iters=passes,  # number of iterations of refinement for each scale
                lr=lr,  # learning rate
                min_side=min_side,  # all sides of image on all scales should be >= min_side / sqrt(2)
                max_scales=max_scales,  # max number of downscaling scales for the image-mask pyramid
                px_budget=px_budget,  # pixels budget. Any image will be resized to satisfy height*width <= px_budget
                modulo=pad_out_to_modulo,  # pad the image to ensure dimension % modulo == 0
                gpu_ids=self.gpu
            )
            inpainted_mat = inpainted_mat[0].permute(1, 2, 0).detach().cpu().numpy()
            self.d_print(f'REFINING on {self.device}')
        else:
            batch = default_collate([dataset])
            with torch.no_grad():
                self.d_print(batch)
                batch = move_to_device(batch, self.device)
                batch['mask'] = (batch['mask'] > 0) * 1
                self.d_print('image shape', batch['image'].shape)
                self.d_print('mask shape', batch['mask'].shape)
                batch = self.model(batch)
                inpainted_mat = convert_to_image(batch['inpainted'][0], un_pad_size)
        inpainted_mat = cv2.cvtColor(inpainted_mat, cv2.COLOR_RGB2BGR)
        if self.show_mats:
            original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)  # noqa
            cv2.imshow('original', original)
            original = visualize_polys(original, polys)  # noqa
            cv2.imshow('poly mask', original)
            cv2.imshow('inpainted', inpainted_mat)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        self.d_print('output image size', inpainted_mat.shape)
        return inpainted_mat


def test(image: str = None, low_clamp: float = 0.1, high_clamp: float = 0.9, mode: MODES = 'scramble', passes: int = 3):
    """
    Test the auto-removal pipeline.
    """
    r = Remover(
        show_mats=True,
        debug=True,
        lama_refine=True
    )
    mat = r.load_image(image)
    r.load_mat(mat, high_clamp=high_clamp, low_clamp=low_clamp, passes=passes, mode=mode)
