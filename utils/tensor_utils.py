import torch
import numpy as np
from scipy.stats import gaussian_kde, norm
from typing import Optional


def to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.cpu().numpy()


def lerp(a, b, t):
    return a * (1 - t) + b * t


def normalize(array: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    array_min = array.min()
    array_range = (array.max() - array_min)
    if array_range == 0.0:
        return np.zeros_like(array) if isinstance(array, np.ndarray) else torch.zeros_like(array)
    return (array - array_min) / array_range


def batch_normalize(array: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    batch_size = array.shape[0]
    elem_dim = len(array.shape) - 1

    if isinstance(array, torch.Tensor):
        flat_array = array.view(batch_size, -1)
        batch_min = flat_array.min(dim=1).values
        batch_max = flat_array.max(dim=1).values
        batch_min = batch_min.view(batch_size, *[1] * elem_dim)
        batch_max = batch_max.view(batch_size, *[1] * elem_dim)
    else:
        flat_array = array.reshape(batch_size, -1)
        batch_min = flat_array.min(axis=1)
        batch_max = flat_array.max(axis=1)
        batch_min = batch_min.reshape(batch_size, *[1] * elem_dim)
        batch_max = batch_max.reshape(batch_size, *[1] * elem_dim)

    return (array - batch_min) / (batch_max - batch_min)


def linear_sample(x: np.ndarray, num=1000) -> np.ndarray:
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    x_range = x_max - x_min
    indices = np.linspace(x_min - 0.5 * x_range, x_max + 0.5 * x_range, num=num)
    return indices


def get_kde(x: np.ndarray, num=1000) -> np.ndarray:
    return gaussian_kde(x).evaluate(points=linear_sample(x, num=num))


def normal_pdf(x: np.ndarray | torch.Tensor,
               loc: np.ndarray | torch.Tensor | int | float,
               scale: np.ndarray | torch.Tensor | int | float
               ) -> np.ndarray | torch.Tensor:
    if scale == 0.0:
        raise ValueError("Scale is zero")

    if not isinstance(x, torch.Tensor):
        return norm.pdf(x, loc, scale)

    loc = torch.as_tensor(loc, dtype=x.dtype)
    scale = torch.as_tensor(scale, dtype=x.dtype)

    denominator = torch.sqrt(2 * torch.pi * scale ** 2)
    power = - torch.pow(x - loc, 2) / (2 * scale ** 2)

    return torch.exp(power) / denominator


# region Gradients
def set_require_grads(inputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
                      requires_grad: bool = True) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]:
    if isinstance(inputs, torch.Tensor):
        if torch.is_floating_point(inputs):
            inputs.requires_grad_(requires_grad=requires_grad)
        return inputs

    elif isinstance(inputs, list):
        return [set_require_grads(tensor) for tensor in inputs]

    elif isinstance(inputs, tuple):
        return tuple(set_require_grads(list(inputs)))

    elif isinstance(inputs, np.ndarray):
        return set_require_grads(torch.as_tensor(inputs))

    else:
        raise NotImplementedError("Type `{}` is not supported (yet).".format(type(inputs)))


def get_gradients(inputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
                  ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None:
    if isinstance(inputs, torch.Tensor):
        if inputs.grad is None:
            return None
        return inputs.grad.data

    elif isinstance(inputs, list):
        return [get_gradients(tensor) for tensor in inputs]

    elif isinstance(inputs, tuple):
        return tuple(get_gradients(list(inputs)))

    else:
        raise NotImplementedError("Type `{}` is not supported (yet).".format(type(inputs)))


# endregion


def to_device(tensor: torch.Tensor | tuple[torch.Tensor] | list[torch.Tensor], device):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    
    elif isinstance(tensor, tuple):
        return tuple([to_device(_tensor, device) for _tensor in tensor])
    
    elif isinstance(tensor, list):
        return [to_device(_tensor, device) for _tensor in tensor]
    
    else:
        raise TypeError("Incorrect type for `tensor`, expected a torch.Tensor, " \
                        "tuple or list, got {}.".format(type(tensor)))

class Range(object):
    def __init__(self, minimum: float, maximum: float):
        self.minimum = minimum
        self.maximum = maximum

    @property
    def delta(self) -> float:
        return self.maximum - self.minimum

    @staticmethod
    def from_str(s: str) -> Optional["Range"]:
        if s == "sigmoid":
            return Range(0.0, 1.0)
        elif s == "tanh":
            return Range(-1.0, 1.0)
        else:
            return None
