"""
Utilities.
"""
import cv2
import numpy as np
from PIL import Image


def convert_to_image(tensor, un_pad_size: int = None) -> np.ndarray:
    """
    Converts a tensor to an image...
    """
    image = tensor.permute(1, 2, 0).detach().cpu().numpy()
    if un_pad_size is not None:
        orig_height, orig_width = un_pad_size
        image = image[:orig_height, :orig_width]
    image = np.clip(image * 255, 0, 255).astype('uint8')
    return image


def overlay_images(background_img: np.ndarray, overlay_img: np.ndarray) -> np.ndarray:
    """
    Combine two images using an alpha channel.
    """
    if overlay_img.shape[2] != 4:
        raise ValueError("Overlay image must have RGBA format (4 channels)")
    if background_img.shape[:2] != overlay_img.shape[:2]:
        overlay_img = cv2.resize(overlay_img, (background_img.shape[1], background_img.shape[0]))
    alpha = overlay_img[:, :, 3] / 255.0
    inv_alpha = 1.0 - alpha
    result = np.empty_like(background_img)
    for c in range(3):  # RGB channels
        result[:, :, c] = (alpha * overlay_img[:, :, c] + inv_alpha * background_img[:, :, c]).astype(np.uint8)
    return result


def expand_and_blur_mask(mask, expand_radius, blur_radius) -> np.ndarray:
    """
    This will dilate the mask by N pixels and blur the mask with a radius of N.
    """
    kernel = np.ones((expand_radius * 2 + 1, expand_radius * 2 + 1), np.uint8)
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)
    blurred_mask = cv2.GaussianBlur(expanded_mask.astype(np.float32), (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
    blurred_mask = np.uint8(blurred_mask)
    return blurred_mask


def create_alpha_channel(image: np.ndarray, alpha: np.ndarray) -> Image:
    """
    Create an RGBA PIL image.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    alpha_resized = cv2.resize(alpha, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    rgba_image = np.zeros((*rgb_image.shape[:2], 4), dtype=np.uint8)
    rgba_image[:, :, :3] = rgb_image
    rgba_image[:, :, 3] = alpha_resized
    return rgba_image


def mask_refine(
        original_image: np.ndarray, inpainted_image: np.ndarray, mask: np.ndarray, radius: int = 5
) -> np.ndarray:
    """
    An attempt at a mask based detail refiner.
    """
    blurred_mask = expand_and_blur_mask(mask, radius, radius * 4)
    max_value = np.iinfo(blurred_mask.dtype).max  # Maximum value for the data type
    blurred_mask = max_value - blurred_mask

    cv2.imshow('blurred_mask', blurred_mask)

    rgba_image = create_alpha_channel(original_image, blurred_mask)
    composite = overlay_images(inpainted_image, rgba_image)
    return composite


def create_opacity_mask(
        heatmap_image: np.ndarray,
        hottest_color: list = [130, 29, 43],  # noqa
        coldest_color: list = [8, 0, 123],  # noqa
        low_clamp: float = 0.0,
        high_clamp: float = 0.0,
        half: bool = False
):
    """
    Convert heatmap to alpha mask and clamp the output.
    """
    if low_clamp >= high_clamp:
        raise ValueError("low clamp cannot be greater than high clamp")
    hottest_color = np.array(hottest_color)
    coldest_color = np.array(coldest_color)
    distances_to_hottest = np.linalg.norm(heatmap_image.astype(np.float32) - hottest_color, axis=-1)
    distances_to_coldest = np.linalg.norm(heatmap_image.astype(np.float32) - coldest_color, axis=-1)
    max_distance_hottest = np.max(distances_to_hottest)
    min_distance_hottest = np.min(distances_to_hottest)
    normalized_distances_hottest = (distances_to_hottest - min_distance_hottest) / (
                max_distance_hottest - min_distance_hottest)

    max_distance_coldest = np.max(distances_to_coldest)
    min_distance_coldest = np.min(distances_to_coldest)
    normalized_distances_coldest = (distances_to_coldest - min_distance_coldest) / (
                max_distance_coldest - min_distance_coldest)
    opacity = (1 - normalized_distances_coldest) * normalized_distances_hottest

    if half:
        height, width = opacity.shape
        half_width = width // 2
        left_half = opacity[:, :half_width]
        right_half = opacity[:, half_width:]
        opacity = left_half + right_half

    opacity = np.clip(opacity, 0, 1)
    if low_clamp:
        opacity[opacity < low_clamp] = 0
    if high_clamp:
        opacity[opacity > high_clamp] = 1
    opacity_mask = (opacity * 255).astype(np.uint8)
    opacity_mask[opacity_mask != 0] = 255
    return opacity_mask


def expand_polygon(_poly: np.ndarray, image_shape: tuple, size: int = 2) -> np.ndarray:
    """
    Enlarge the poly by N pix in all directions.
    """
    min_x, min_y = np.min(_poly, axis=0)
    max_x, max_y = np.max(_poly, axis=0)
    min_x -= size
    min_y -= size
    max_x += size
    max_y += size
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(image_shape[1] - 1, max_x)
    max_y = min(image_shape[0] - 1, max_y)
    expanded_poly = np.array([[min_x, min_y],
                              [max_x, min_y],
                              [max_x, max_y],
                              [min_x, max_y]], dtype=np.float32)
    return expanded_poly


def mask_from_polys(mask: np.ndarray, polys: np.ndarray) -> np.ndarray:
    """
    Just draw a buncha squares...
    """
    void_mask = np.zeros_like(mask, dtype=np.uint8)
    for poly in polys:
        poly = expand_polygon(poly, mask.shape, 15)
        poly = np.int32([poly])  # Convert polygon to integer.
        cv2.fillPoly(void_mask, poly, (255, 255, 255))
    return void_mask


def fetch_contours(image: np.array, mask: np.array, polygons: np.array, show_images: bool = False):
    """
    Fetch text contours from within the polys

    Whilst this is mega cool, it has no effect on the model's evaluation of the text and just adds processing...
    """
    def find_contours(_image: np.array) -> tuple:
        """
        Locates the boundaries of the edges within the image.
        """
        _contours, _hierarchy = cv2.findContours(
            _image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return _contours, _hierarchy

    image = np.array(image)  # Create new instance for viewing.
    void_mask = np.zeros_like(mask, dtype=np.uint8)
    for poly in polygons:
        poly = expand_polygon(poly, image.shape)
        poly = np.int32([poly])  # Convert polygon to integer.
        poly_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)  # Mask for the current polygon.
        cv2.fillPoly(poly_mask, poly, 255)  # noqa
        region_of_interest = cv2.bitwise_and(image, image, mask=poly_mask)  # Bitwise AND to get the region of interest.
        edges = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)  # Convert the region of interest to grayscale.
        edges = cv2.Canny(edges, 50, 150)
        cv2.polylines(edges, [poly.astype(np.int32)], isClosed=True, color=(0, 0, 0), thickness=2)
        if show_images:
            cv2.imshow('gray mask step 1', edges)
            cv2.waitKey(0)

        contours, _ = find_contours(edges)  # Find contours pass 1.
        cv2.drawContours(edges, contours, -1, (255, 255, 255), 2)
        if show_images:
            cv2.imshow('gray mask step 2', edges)
            cv2.waitKey(0)

        contours, _ = find_contours(edges)  # Find contours pass 2.
        cv2.fillPoly(edges, contours, (255, 255, 255))
        if show_images:
            cv2.imshow('gray mask step 3', edges)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.drawContours(image, contours, -1, (0, 0, 255), cv2.FILLED)  # Draw contours on the original image.
            cv2.fillPoly(image, contours, (255, 0, 0))
        cv2.drawContours(void_mask, contours, -1, (255, 255, 255), cv2.FILLED)  # Draw contours on the mask.
        cv2.fillPoly(void_mask, contours, (255, 255, 255))

    exclusion_mask = np.array(mask)
    exclusion_mask[exclusion_mask != 0] = 255
    exclusion_mask = 255 - exclusion_mask
    combined_mask = void_mask - exclusion_mask
    regions = combined_mask == 255
    final_mask = np.array(mask)
    final_mask[regions] = combined_mask[regions]
    return final_mask


def visualize_polys(mat: np.array, polys: np.array, color: tuple = (0, 255, 0), thickness: int = 1) -> np.ndarray:
    """
    This will draw the polygon lines into an image mat matching the size of the original.
    """
    shape = mat.shape
    if len(shape) == 2 or shape[-1] == 1:
        mat = np.stack((mat, mat, mat), axis=-1)

    for poly in polys:
        cv2.polylines(mat, [poly.astype(np.int32)], isClosed=True, color=color, thickness=thickness)
    return mat
