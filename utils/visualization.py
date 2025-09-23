import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
from typing import Sequence

from utils.tensor_utils import to_numpy
from utils.tensor_utils import normalize as _normalize
from utils.imaging import upscale_resolution


# region Curves
def _prepare_curves(curves: torch.Tensor | np.ndarray | list[torch.Tensor],
                    normalize: bool) -> list[np.ndarray]:
    if not isinstance(curves, (tuple, list)):
        curves = [curves]

    curves = [to_numpy(curve) for curve in curves]

    if normalize:
        curves = [_normalize(curve) for curve in curves]

    return curves


def plot_line2d_to_array(x: torch.Tensor | np.ndarray | list[torch.Tensor],
                         y: torch.Tensor | np.ndarray | list[torch.Tensor],
                         output_size: tuple[int, int] = None,
                         dpi=75.0,
                         normalize=True,
                         use_xy_limits=True,
                         curves_names: list[str] = None) -> np.ndarray:
    x = _prepare_curves(x, normalize)
    y = _prepare_curves(y, normalize)

    if curves_names is None:
        curves_names = list(range(len(x)))

    if output_size is None:
        figsize = plt.rcParams["figure.figsize"]
        output_size = (int(figsize[1] * dpi), int(figsize[0] * dpi))
    else:
        figsize = [output_size[1] / dpi, output_size[0] / dpi]

    figure = plt.Figure(figsize=figsize, dpi=dpi)
    plot = figure.add_subplot(111)
    if use_xy_limits:
        plot.set_xlim(0.0, 1.0)
        plot.set_ylim(0.0, 1.0)

    for x_curve, y_curve in zip(x, y):
        plot.plot(x_curve, y_curve)

    plot.legend(curves_names)

    canvas = FigureCanvasAgg(figure)
    canvas.draw()

    # noinspection PyTypeChecker
    canvas_as_str: str = canvas.tostring_rgb()
    image = np.fromstring(canvas_as_str, dtype="uint8")
    image = np.reshape(image, [output_size[0], output_size[1], 3])

    return image


# endregion

# region Images

def format_image(image: np.ndarray,
                 color_map=None,
                 normalize_image=True,
                 target_resolution: int | None = None,
                 resize_method=cv2.INTER_NEAREST,
                 rotate: bool = False
                 ) -> np.ndarray:
    if normalize_image:
        image = _normalize(image)

    if (len(image.shape) != 2) and ((len(image.shape) != 3) and (image.shape[-1] != 1)):
        raise ValueError("Incorrect image shape: {}".format(image.shape))

    if rotate:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image = (image * 255.0).astype(np.uint8)

    if color_map is not None:
        image = cv2.applyColorMap(image, color_map)

    if target_resolution is not None:
        # noinspection PyTypeChecker
        target_size, _ = upscale_resolution(image.shape[:2], max_target=target_resolution)
        image = cv2.resize(image, tuple(reversed(target_size)), interpolation=resize_method)
        cv2.waitKey(0)

    return image


# region Image interpretability


def compute_iso_lines(intensity: np.ndarray,
                      iso_percentiles: Sequence[int] = (25, 50, 75, 90),
                      color_map: int = cv2.COLORMAP_JET,
                      ) -> tuple[np.ndarray, np.ndarray]:
    if intensity.ndim == 2:
        intensity = np.expand_dims(intensity, axis=-1)

    iso_th = [(intensity.max() * i / 100).astype(intensity.dtype) for i in iso_percentiles]
    iso_masks = [np.float32(intensity > th) for th in iso_th]
    iso_mask = np.zeros_like(iso_masks[0])

    for i in range(len(iso_th)):
        iso_mask_blurred = cv2.blur(iso_masks[i], ksize=(4, 4))
        iso_mask_blurred = np.expand_dims(iso_mask_blurred, axis=-1)
        iso_line = (iso_mask_blurred - iso_masks[i]) > 0
        iso_mask = np.maximum(iso_mask, np.float32(iso_line) * iso_th[i])

    iso_lines = cv2.applyColorMap(np.uint8(iso_mask), color_map)
    iso_mask = np.float32(iso_mask > 0)

    return iso_lines, iso_mask


def add_iso_lines(image: np.ndarray,
                  intensity: np.ndarray,
                  iso_percentiles: Sequence[int] = (25, 50, 75, 90),
                  color_map: int = cv2.COLORMAP_JET,
                  ) -> np.ndarray:
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    if intensity.ndim == 2:
        intensity = np.expand_dims(intensity, axis=-1)

    iso_lines, iso_mask = compute_iso_lines(intensity, iso_percentiles, color_map)
    image = image * (1.0 - iso_mask) + iso_lines * iso_mask
    return image


def overlay_image(image: np.ndarray,
                  intensity: np.ndarray,
                  color_map: int = cv2.COLORMAP_JET,
                  mask_threshold: float | None = None,
                  iso_percentiles: Sequence[int] | None = (25, 50, 75, 90),
                  overlay_coeff: float = 0.2,
                  ) -> np.ndarray:
    # region Normalize images
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    image = _normalize(np.float32(image)) * 255.0

    if intensity.ndim == 2:
        intensity = np.expand_dims(intensity, axis=-1)
    intensity = _normalize(np.float32(intensity)) * 255.0
    # endregion

    # region Compute overlay mask (from intensity)
    overlay_mask = intensity
    if mask_threshold is None:
        min_percent = 25 if iso_percentiles is None else min(iso_percentiles)
        mask_threshold = (overlay_mask.max() * min_percent / 100).astype(overlay_mask.dtype)

    overlay_mask = overlay_mask > mask_threshold
    # endregion

    # region Overlay on top of image using mask
    intensity_rgb = np.float32(cv2.applyColorMap(np.uint8(intensity), color_map))

    masked_image = intensity_rgb * overlay_coeff + image * (1.0 - overlay_coeff)
    overlaid_image = image * (1.0 - overlay_mask) + masked_image * overlay_mask
    # endregion

    # region (Optional) Add iso lines
    if iso_percentiles is not None:
        overlaid_image = add_iso_lines(overlaid_image, intensity, iso_percentiles, color_map)
    # endregion

    return overlaid_image


# endregion

# region Stitch images
def stitch_images(images: list[np.ndarray]) -> np.ndarray:
    col_count = int(np.ceil(np.sqrt(len(images))))
    image_width, image_height, *other_dims = images[0].shape

    result = np.zeros(shape=(image_width * col_count,
                             image_height * col_count,
                             *other_dims),
                      dtype=images[0].dtype)

    for row_index in range(col_count):
        for col_index in range(col_count):
            image_index = col_index + row_index * col_count
            if image_index >= len(images):
                continue

            start_col, start_row = col_index * image_width, row_index * image_height
            end_col, end_row = start_col + image_width, start_row + image_height
            result[start_col:end_col, start_row:end_row] = images[image_index]

    for row_index in range(col_count):
        start_row = row_index * image_height
        end_row = start_row + image_height

        result[:, start_row, :, :] = 1.0
        result[:, end_row - 1, :, :] = 1.0

    for col_index in range(col_count):
        start_col = col_index * image_width
        end_col = start_col + image_width

        result[start_col, :, :, :] = 1.0
        result[end_col - 1, :, :, :] = 1.0

    return result


def show_stitched_images(images: list[np.ndarray]):
    image = stitch_images(images)
    image = np.transpose(image, axes=(1, 0, 2, 3))
    i = image.shape[2] // 4
    key = -1
    while key == -1:
        image_slice = image[..., i, :]
        cv2.imshow("image", image_slice)
        key = cv2.waitKey(50)
        i = (i + 1) % image.shape[2]

# endregion

# endregion
