import torch
from torch.nn.functional import conv3d
from torchvision.transforms.functional import convert_image_dtype
# noinspection PyProtectedMember
from torchvision.transforms._functional_tensor import _scale_channel
import numpy as np
import cv2
from pathlib import Path
from typing import Sequence, Literal

from utils.tensor_utils import lerp


# region Blurring (Gaussian / Flat)
def index_to_gaussian_dist(index: int, size: int):
    coord = index - (size - 1) // 2
    return coord * coord


def get_3d_gaussian_kernel(size: int, sigma=1.0) -> torch.Tensor:
    kernel = np.zeros(shape=[size, size, size], dtype=np.float32)
    constant_factor = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    constant_power = - 1.0 / (2 * sigma * sigma)
    for i in range(size):
        x = index_to_gaussian_dist(i, size)
        for j in range(size):
            y = index_to_gaussian_dist(j, size)
            for k in range(size):
                z = index_to_gaussian_dist(k, size)
                kernel[i, j, k] = constant_factor * np.exp((x + y + z) * constant_power)

    kernel = torch.as_tensor(kernel, dtype=torch.float32)
    kernel = torch.reshape(kernel, [1, 1, size, size, size])
    return kernel


def get_3d_flat_blur_kernel(size: int = 3,
                            center_weight: float = None,
                            channels: int = 1,
                            ) -> torch.Tensor:
    center_index = size // 2
    if center_weight is None:
        center_weight = 5.0 / 9.0 * (size ** 3)

    kernel = torch.ones(size=[size, size, size], dtype=torch.float32)
    kernel[center_index, center_index, center_index] = center_weight
    kernel /= kernel.sum()
    kernel = kernel.expand(channels, 1, size, size, size)

    return kernel


def blur_image_3d(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    return conv3d(image, kernel, padding="same")


def blur_image_3d_repeat(image: torch.Tensor, kernel: torch.Tensor, n: int) -> torch.Tensor:
    for _ in range(n):
        image = blur_image_3d(image, kernel)
    return image


def blur_image_3d_flat_kernel(image: torch.Tensor,
                              kernel_size: int = 3,
                              kernel_center_weight: float = None):
    if len(image.shape) == 3:
        channels = 1
        image = image.unsqueeze(0)
        requires_squeeze = True
    else:
        channels = image.size(-4)
        requires_squeeze = False

    kernel = get_3d_flat_blur_kernel(kernel_size, kernel_center_weight, channels)
    image_blurred = conv3d(image, kernel, groups=channels)

    pad_size_l = pad_size_r = kernel_size // 2
    if kernel_size % 2 == 0:
        pad_size_l -= 1
    image = image.clone()
    image[..., pad_size_l:-pad_size_r, pad_size_l:-pad_size_r, pad_size_l:-pad_size_r] = image_blurred

    if requires_squeeze:
        image = image.squeeze(0)

    return image


# endregion

# region Transforms
def get_image_max_bound(image: torch.Tensor):
    return 1.0 if image.is_floating_point() else 255.0


def adjust_brightness(image: torch.Tensor, factor: float) -> torch.Tensor:
    dtype = image.dtype
    image = lerp(torch.zeros_like(image), image, factor)
    max_bound = get_image_max_bound(image)
    return image.clamp(0, max_bound).to(dtype)


def adjust_contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    dtype = image.dtype if torch.is_floating_point(image) else torch.float32
    dims = list(range(len(image.shape)))
    mean = torch.mean(image.to(dtype), keepdim=True, dim=dims)
    return lerp(image, mean, factor)


def adjust_sharpness_3d(image: torch.Tensor, factor: float) -> torch.Tensor:
    blurred_image = blur_image_3d_flat_kernel(image)
    return lerp(image, blurred_image, factor)


def posterize(image: torch.Tensor, bits: int) -> torch.Tensor:
    original_dtype = image.dtype
    if original_dtype != torch.uint8:
        image = convert_image_dtype(image, dtype=torch.uint8)

    mask = -int(2 ** (8 - bits))
    image = image & mask

    if original_dtype != torch.uint8:
        image = convert_image_dtype(image, dtype=original_dtype)
    return image


def invert(image: torch.Tensor) -> torch.Tensor:
    max_bound = get_image_max_bound(image)
    max_bound = torch.tensor(max_bound, dtype=image.dtype, device=image.device)
    return max_bound - image


def solarize(image: torch.Tensor, threshold: float | int) -> torch.Tensor:
    inverted_image = invert(image)
    return torch.where(image >= threshold, inverted_image, image)


def autocontrast_3d(image: torch.Tensor) -> torch.Tensor:
    max_bound = get_image_max_bound(image)
    target_dtype = image.dtype if torch.is_floating_point(image) else torch.float32

    minimum = image.amin(dim=(-3, -2, -1), keepdim=True).to(target_dtype)
    maximum = image.amax(dim=(-3, -2, -1), keepdim=True).to(target_dtype)
    scale = max_bound / (maximum - minimum)
    indices = torch.isfinite(scale).logical_not()
    minimum[indices] = 0
    scale[indices] = 1

    return ((image - minimum) * scale).clamp(0, max_bound).to(image.dtype)


def equalize_3d(image: torch.Tensor) -> torch.Tensor:
    original_dtype = image.dtype
    if original_dtype != torch.uint8:
        image = convert_image_dtype(image, dtype=torch.uint8)

    requires_squeeze = False
    if image.ndim == 3:
        image = image.unsqueeze(dim=0)
        requires_squeeze = True
    elif image.ndim >= 5:
        raise NotImplementedError("Batched images are not supported yet.")

    image = [_scale_channel(image[channel_index]) for channel_index in range(image.size(0))]
    image = torch.stack(image, dim=0)

    if requires_squeeze:
        image = image.squeeze(dim=0)

    if original_dtype != torch.uint8:
        image = convert_image_dtype(image, dtype=original_dtype)

    return image


# endregion

# region Slices
def get_3d_image_slices(image: np.ndarray | torch.Tensor,
                        method: Literal["center", "max_intensity"],
                        add_red_lines=True,
                        target_resolution: int | None = 512,
                        dims=(0, 1, 2)
                        ) -> list[np.ndarray]:
    image: np.ndarray = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
    image = remove_image_extra_channel(image)

    slices_indices = get_3d_image_slices_indices(image, method)
    # noinspection PyTypeChecker

    target_resolution, upscale_ratio = upscale_resolution(image.shape[:3], max_target=target_resolution)

    slices = []
    for dim in dims:
        new_size = get_slice_resize_shape(target_resolution, dim)

        if add_red_lines:
            red_lines_position = [int(slices_indices[i] * upscale_ratio) for i in range(3) if i != dim]
        else:
            red_lines_position = None

        image_slice = get_formatted_3d_image_slice(image, dim, new_size, slices_indices[dim], red_lines_position)
        slices.append(image_slice)

    return slices


def get_formatted_3d_image_slice(image: np.ndarray | torch.Tensor,
                                 dim: int,
                                 new_size: tuple[int, int],
                                 slice_index: int | None = None,
                                 red_lines_position: tuple[int, int] | list[int] | None = None,
                                 ) -> np.ndarray:
    """
    Returns a formatted slice from the 3D image at the given dimension (dim) and given size (new_size).
    :param image: The 3D image the slice is extracted from.
    :param dim: The reduced spatial dimension.
    :param new_size: The 2D size of the formatted output.
    :param slice_index: Optional. When not provided, the slice the maximum average intensity is selected.
    :param red_lines_position: Optional. When provided, a horizontal line and a vertical line are drawn are the given
        indices.
    :return: A formatted 2D image slice.
    """
    image: np.ndarray = image.numpy() if isinstance(image, torch.Tensor) else image
    image = remove_image_extra_channel(image)

    if slice_index is None:
        slice_index = get_3d_image_max_intensity_slice_index(image, dim)

    image_slice = get_3d_image_slice(image, dim, slice_index)
    image_slice = cv2.resize(image_slice, tuple(reversed(new_size)), interpolation=cv2.INTER_NEAREST)
    if (len(image_slice.shape) == 2) or (image_slice.shape[-1] == 1):
        image_slice = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2BGR)

    if red_lines_position is not None:
        x, y = red_lines_position
        image_slice[x, :, 2] = 1.0
        image_slice[:, y, 2] = 1.0

    image_slice = cv2.rotate(image_slice, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image_slice


def get_3d_image_slices_indices(image: np.ndarray,
                                method: Literal["center", "max_intensity"],
                                ) -> tuple[int, int, int]:
    if image.shape[0] <= 3:
        raise NotImplementedError("Image shape {} started with channels, "
                                  "which is not supported yet".format(image.shape))

    if method == "center":
        # noinspection PyTypeChecker
        return tuple([dim // 2 for dim in image.shape[:3]])
    elif method == "max_intensity":
        # noinspection PyTypeChecker
        return tuple([get_3d_image_max_intensity_slice_index(image, dim) for dim in range(3)])
    else:
        raise ValueError("Method `{}` is not recognized.".format(method))


def get_3d_image_max_intensity_slice_index(image: np.ndarray, dim: int) -> int:
    if len(image.shape) == 4:
        image = image.mean(axis=-1)

    if dim < 0:
        dim = len(image.shape) - dim

    reduction_axes = tuple([i for i in range(3) if i != dim])
    return int(np.argmax(np.sum(image, axis=reduction_axes)))


def get_3d_image_slice(image: np.ndarray, dim: int, index: int):
    if dim < 0:
        dim = len(image.shape) - dim

    if dim == 0:
        return image[index]
    elif dim == 1:
        return image[:, index]
    elif dim == 2:
        return image[:, :, index]
    else:
        raise ValueError(dim)


def upscale_resolution(base_shape: tuple[int, int, int],
                       max_target: int
                       ) -> tuple[tuple[int, int, int], float]:
    max_base = max(base_shape)
    ratio = max_target / max_base
    # noinspection PyTypeChecker
    return tuple([int(x * ratio) for x in base_shape]), ratio


def get_slice_resize_shape(base_shape: tuple[int, int, int],
                           dim: int
                           ) -> tuple[int, int]:
    # noinspection PyTypeChecker
    return tuple([base_shape[i] for i in range(len(base_shape)) if i != dim])


def write_slices(image: torch.Tensor | np.ndarray,
                 image_name: str,
                 output_folder: Path,
                 method: Literal["center", "max_intensity"]
                 ) -> None:
    slices = get_3d_image_slices(image, method, add_red_lines=False)
    for dim, sample_slice in enumerate(slices):
        output_path = output_folder / "{}_dim-{}.png".format(image_name, dim)
        cv2.imwrite(output_path.as_posix(), sample_slice * 255.0)


# endregion

def get_center_of_mass(image: torch.Tensor, exponent=1.0) -> tuple[int, ...]:
    coords = [np.arange(size) for size in image.shape]
    coords = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1)
    coords = np.reshape(coords, [-1, len(image.shape)])
    weights = image.reshape([-1, 1])
    weights = weights - weights.min()
    if exponent != 1.0:
        weights = weights.pow(exponent=exponent)
    weights = weights / weights.sum()

    center_of_mass = np.round((coords * weights).sum(axis=0))
    # noinspection PyTypeChecker
    return tuple(int(x) for x in center_of_mass)


def remove_image_extra_channel(image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if image.shape[0] == 1:
        if isinstance(image, torch.Tensor):
            image = torch.squeeze(image, dim=0)
        else:
            image = np.squeeze(image, axis=0)

    if image.shape[-1] == 1:
        if isinstance(image, torch.Tensor):
            image = torch.squeeze(image, dim=-1)
        else:
            image = np.squeeze(image, axis=-1)

    return image


def grayscale_to_rgb(image: torch.Tensor, channels_dim: int = 1) -> torch.Tensor:
    if channels_dim < 0:
        channels_dim = len(image.shape) + channels_dim

    repeats = [3 if (i == channels_dim) else 1 for i in range(len(image.shape))]
    return image.repeat(*repeats)


def index_to_tricolor(image: torch.Tensor, channels_dims: int = 1, ignore_background=True) -> torch.Tensor:
    image = image.to(torch.int64)
    offset = 1 if ignore_background else 0
    rgb = [(image == (i + offset)) for i in range(3)]
    # noinspection PyTypeChecker
    image = torch.concat(rgb, dim=channels_dims)
    return image


def apply_palette(image: torch.Tensor, channels_dim: int = 1, ignore_background=True) -> torch.Tensor:
    import seaborn as sns

    image = image.to(torch.int64)
    device = image.device

    # noinspection PyTypeChecker
    channels_dim = image.shape + channels_dim if channels_dim < 0 else channels_dim
    output_shape = [dim if i != channels_dim else 3 for i, dim in enumerate(image.shape)]
    color_shape = [1 if i != channels_dim else 3 for i in range(len(image.shape))]
    dtype = torch.float32
    output = torch.zeros(size=output_shape, device=device, dtype=dtype)

    offset = 1 if ignore_background else 0
    color_count = int(image.max()) + offset
    palette = sns.color_palette("husl", color_count)
    palette = [torch.as_tensor(color, dtype=dtype, device=device).reshape(color_shape) for color in palette]
    unique_indices = image.unique()

    for i in range(color_count):
        if i not in unique_indices:
            continue

        index = i + offset
        output += (image == index).to(dtype) * palette[i]

    return output


def spacing_to_affine(spacing: Sequence[float], device=None) -> torch.Tensor:
    affine = torch.eye(n=4, dtype=torch.float32, device=device)
    for i in range(3):
        affine[i][i] = float(spacing[i])
    return affine
