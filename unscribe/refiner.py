import torch
import torch.nn as nn

from unscribe.saicinpainting.evaluation.data import pad_tensor_to_modulo
from unscribe.saicinpainting.evaluation.utils import move_to_device
from unscribe.saicinpainting.training.modules.ffc import FFCResnetBlock
from unscribe.saicinpainting.training.modules.pix2pixhd import ResnetBlock
from unscribe.saicinpainting.evaluation.refinement import (
    _infer,  # noqa
    _get_image_mask_pyramid,  # noqa

)


def refine_predict(
        batch: dict, inpainter: nn.Module, gpu_ids: str,
        modulo: int, n_iters: int, lr: float, min_side: int,
        max_scales: int, px_budget: int
):
    """Refines the inpainting of the network

    Parameters
    ----------
    batch : dict
        image-mask batch, currently we assume the batch size to be 1
    inpainter : nn.Module
        the inpainting neural network
    gpu_ids : str
        the GPU ids of the machine to use. If only single GPU, use: "0,"
    modulo : int
        pad the image to ensure dimension % modulo == 0
    n_iters : int
        number of iterations of refinement for each scale
    lr : float
        learning rate
    min_side : int
        all sides of image on all scales should be >= min_side / sqrt(2)
    max_scales : int
        max number of downscaling scales for the image-mask pyramid
    px_budget : int
        pixels budget. Any image will be resized to satisfy height*width <= px_budget

    Returns
    -------
    torch.Tensor
        inpainted image of size (1,3,H,W)
    """

    assert not inpainter.training
    assert not inpainter.add_noise_kwargs
    assert inpainter.concat_mask
    if gpu_ids == 'cuda':
        gpu_ids = '0'
    gpu_ids = [f'cuda:{gpuid}' for gpuid in gpu_ids.replace(" ", "").split(",") if gpuid.isdigit()]
    if not gpu_ids:
        gpu_ids = ['cpu']
    n_resnet_blocks = 0  # noqa
    first_resblock_ind = 0
    found_first_resblock = False
    for idl in range(len(inpainter.generator.model)):
        if isinstance(inpainter.generator.model[idl], FFCResnetBlock) or isinstance(inpainter.generator.model[idl],
                                                                                    ResnetBlock):
            n_resnet_blocks += 1
            found_first_resblock = True
        elif not found_first_resblock:
            first_resblock_ind += 1
    resblocks_per_gpu = n_resnet_blocks // len(gpu_ids)

    if gpu_ids != ['cpu']:
        devices = [torch.device(gpu_id) for gpu_id in gpu_ids]
    else:
        devices = [torch.device('cpu')]

    # split the model into front, and rear parts
    forward_front = inpainter.generator.model[0:first_resblock_ind]  # noqa
    forward_front.to(devices[0])
    forward_rears = []
    for idd in range(len(gpu_ids)):
        if idd < len(gpu_ids) - 1:
            forward_rears.append(inpainter.generator.model[first_resblock_ind + resblocks_per_gpu * (
                idd):first_resblock_ind + resblocks_per_gpu * (idd + 1)])
        else:
            forward_rears.append(inpainter.generator.model[first_resblock_ind + resblocks_per_gpu * (idd):])  # noqa
        forward_rears[idd].to(devices[idd])

    ls_images, ls_masks = _get_image_mask_pyramid(
        batch,
        min_side,
        max_scales,
        px_budget
    )
    image_inpainted = None

    for ids, (image, mask) in enumerate(zip(ls_images, ls_masks)):
        orig_shape = image.shape[2:]
        image = pad_tensor_to_modulo(image, modulo)
        mask = pad_tensor_to_modulo(mask, modulo)
        mask[mask >= 1e-8] = 1.0
        mask[mask < 1e-8] = 0.0
        image, mask = move_to_device(image, devices[0]), move_to_device(mask, devices[0])
        if image_inpainted is not None:
            image_inpainted = move_to_device(image_inpainted, devices[-1])
        image_inpainted = _infer(
            image, mask, forward_front, forward_rears,  # noqa
            image_inpainted, orig_shape, devices, ids,
                                 n_iters, lr
        )
        image_inpainted = image_inpainted[:, :, :orig_shape[0], :orig_shape[1]]
        # detach everything to save resources
        image = image.detach().cpu()  # noqa
        mask = mask.detach().cpu()  # noqa
    return image_inpainted
