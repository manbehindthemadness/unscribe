"""
Hold our general logic.
"""
import os
import cv2
import numpy as np
import torch
import yaml
import PIL.Image as Image
from pathlib import Path
from mcraft import TNet
from omegaconf import OmegaConf
from dscribe.saicinpainting.evaluation.utils import move_to_device
from torch.utils.data._utils.collate import default_collate  # noqa
from dscribe.saicinpainting.training.trainers import load_checkpoint
from dscribe.saicinpainting.evaluation.data import scale_image, pad_img_to_modulo
try:
    from utils import (
        create_opacity_mask
    )
except ImportError:
    from .utils import (
        create_opacity_mask
    )

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

models_path = Path('~/.dscribe/').expanduser()


class Remover(TNet):
    """
    Removes text and watermarks.
    """
    def __init__(
            self,
            cuda: bool = False,
            poly: bool = False,
            refine: bool = False,
    ):
        super().__init__(
            cuda=cuda,
            poly=poly,
            refine=refine
        )
        self.device = torch.device("cuda:0" if cuda else "cpu")
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
        )
        self.model.freeze()

    @staticmethod
    def process_mat(mat: np.ndarray) -> np.ndarray:
        """
        THis is an adaptation of from dscribe.saicinpainting.evaluation.data.load_image
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
            test_run: bool = False
    ) -> np.ndarray:
        """
        Load and process an image mat.
        """
        if test_run:
            rgb_mat, mask_mat = [
                np.array(Image.open('dscribe/1.png').convert('RGB')),
                np.array(Image.open('dscribe/2.png').convert('L'))
                ]
        else:
            rgb_mat = mat
            _, _, mask_mat = self.forward(mat)  # bboxes, polys, score_text

            mask_mat = create_opacity_mask(mask_mat, half=True, clamp=0.1)
            new_height, new_width = rgb_mat.shape[:2]
            mask_mat = cv2.resize(mask_mat, (new_width, new_height))
            # mask_mat = cv2.equalizeHist(mask_mat)

            cv2.imshow('rgb', rgb_mat)
            cv2.imshow('mask', mask_mat)
            # cv2.waitKey(0)
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
        batch = default_collate([dataset])
        with torch.no_grad():
            print(batch)
            batch = move_to_device(batch, self.device)
            batch['mask'] = (batch['mask'] > 0) * 1
            print('image shape', batch['image'].shape)
            print('mask shape', batch['mask'].shape)
            batch = self.model(batch)
            cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
            if un_pad_size is not None:
                orig_height, orig_width = un_pad_size
                cur_res = cur_res[:orig_height, :orig_width]
        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imshow('inpainted', cur_res)
        original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        cv2.imshow('original', original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return cur_res


def test(image: str = None):
    """
    Test the auto-removal pipeline.
    """
    if not image:
        imsge = 'cover.jpg'
    r = Remover()
    mat = r.load_image(image)
    r.load_mat(mat)

