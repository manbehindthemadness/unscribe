"""
Utilities.
"""
import numpy as np


def create_opacity_mask(
        heatmap_image: np.ndarray,
        hottest_color: list = [130, 29, 43],  # noqa
        coldest_color: list = [8, 0, 123],  # noqa
        clamp: float = 0,
        half: bool = False
):
    """
    Convert heatmap to alpha mask.
    """
    hottest_color = np.array(hottest_color)
    coldest_color = np.array(coldest_color)
    distances_to_hottest = np.linalg.norm(heatmap_image.astype(np.float32) - hottest_color, axis=-1)
    # Calculate the Euclidean distance between each pixel's color and the coldest color
    distances_to_coldest = np.linalg.norm(heatmap_image.astype(np.float32) - coldest_color, axis=-1)

    # Normalize the distances to the range [0, 1] for both hottest and coldest colors
    max_distance_hottest = np.max(distances_to_hottest)
    min_distance_hottest = np.min(distances_to_hottest)
    normalized_distances_hottest = (distances_to_hottest - min_distance_hottest) / (
                max_distance_hottest - min_distance_hottest)

    max_distance_coldest = np.max(distances_to_coldest)
    min_distance_coldest = np.min(distances_to_coldest)
    normalized_distances_coldest = (distances_to_coldest - min_distance_coldest) / (
                max_distance_coldest - min_distance_coldest)

    # Calculate opacity based on both distances
    opacity = (1 - normalized_distances_coldest) * normalized_distances_hottest

    if half:
        height, width = opacity.shape
        half_width = width // 2
        left_half = opacity[:, :half_width]
        right_half = opacity[:, half_width:]
        opacity = left_half + right_half

    # Clamp opacity values to [0, 1]
    opacity = np.clip(opacity, 0, 1)
    if clamp:
        opacity[opacity < clamp] = 0

    # Create a single-channel opacity mask
    opacity_mask = (opacity * 255).astype(np.uint8)
    return opacity_mask


