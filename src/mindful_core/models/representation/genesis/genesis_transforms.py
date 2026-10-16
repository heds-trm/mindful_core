import torch
import torch.nn as nn

from mindful_core.data.transforms.imaging import (
    get_random_block_coordinates, replace_block,
    random_apply, slice_select,
    RandomNDLocalShuffle
)


class RandomNDFlip(nn.Module):
    def __init__(self, p=0.5):
        super(RandomNDFlip, self).__init__()
        self.p = p

    def forward(self, inputs):
        return random_apply(inputs, self.get_flipped, self.p)

    def get_flipped(self, inputs):
        flipped_dims = []
        spatial_dims = len(inputs.shape) - 1
        for dim in range(spatial_dims):
            if torch.rand(1) < self.p:
                flipped_dims.append(dim)
        if len(flipped_dims) > 0:
            inputs = torch.flip(inputs, dims=flipped_dims)
        return inputs


# region Non-linear intensity transform helpers
def comb(n: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return ((n + 1).lgamma() - (k + 1).lgamma() - (n - k + 1).lgamma()).exp()


def bernstein_poly(indices: torch.Tensor, n: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
    return comb(n, indices) * factors ** (n - indices) * (1 - factors) ** indices


def get_bezier_curve(control_points: torch.Tensor, step_count: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    control_points_count = control_points.size(0)
    x, y = control_points[:, 0], control_points[:, 1]

    indices = torch.arange(control_points_count, dtype=torch.float32, device=control_points.device).unsqueeze(1)
    factors = torch.linspace(0.0, 1.0, steps=step_count, device=control_points.device)
    control_points_count = torch.as_tensor(control_points_count, dtype=torch.float32, device=control_points.device)

    polynomial_array = bernstein_poly(indices, control_points_count - 1, factors)
    x = x @ polynomial_array
    y = y @ polynomial_array
    return x, y


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    indices = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indices = torch.clamp(indices, 0, len(m) - 1)

    return m[indices] * x + b[indices]


# endregion

class RandomNonLinearTransformation(nn.Module):
    def __init__(self, p=0.5):
        super(RandomNonLinearTransformation, self).__init__()
        self.p = p

    def forward(self, inputs):
        return random_apply(inputs, self.get_transformed, self.p)

    @staticmethod
    def get_transformed(inputs: torch.Tensor) -> torch.Tensor:
        inputs_shape = inputs.shape

        curve_points = torch.rand(4, 2, device=inputs.device)
        curve_points[0, :] = 0.0
        curve_points[-1, :] = 1.0

        x, y = get_bezier_curve(curve_points, step_count=1000)

        x, _ = torch.sort(x, dim=-1)
        sort_y = torch.rand(1) < 0.5
        if sort_y:
            y, _ = torch.sort(y, dim=-1)

        outputs = interp(inputs.reshape(-1), x, y)
        outputs = outputs.reshape(inputs_shape)
        return outputs


class RandomInOutPainting(nn.Module):
    def __init__(self,
                 p=0.9,
                 in_rate=0.2,
                 inpaint_max_count=5,
                 outpaint_max_count=5,
                 paint_iter_prob=0.95,
                 inpaint_min_ratio=0.15,
                 inpaint_max_ratio=0.35,
                 outpaint_min_ratio=0.45,
                 outpaint_max_ratio=0.60,
                 ):
        super(RandomInOutPainting, self).__init__()
        self.p = p
        self.in_rate = in_rate
        self.inpaint_max_count = inpaint_max_count
        self.outpaint_max_count = outpaint_max_count
        self.paint_iter_prob = paint_iter_prob
        self.inpaint_min_ratio = inpaint_min_ratio
        self.inpaint_max_ratio = inpaint_max_ratio
        self.outpaint_min_ratio = outpaint_min_ratio
        self.outpaint_max_ratio = outpaint_max_ratio

    def forward(self, inputs) -> torch.Tensor:
        return random_apply(inputs, self.get_painted, self.p)

    def get_painted(self, inputs: torch.Tensor) -> torch.Tensor:
        use_inpaint = torch.rand(1) < self.in_rate
        outputs = self.get_inpainted(inputs) if use_inpaint else self.get_outpainted(inputs)
        return outputs

    def get_inpainted(self, inputs: torch.Tensor) -> torch.Tensor:
        spatial_dims = len(inputs.shape) - 1
        outputs = inputs

        i = 0
        stop_painting = False
        while (i < self.inpaint_max_count) and not stop_painting:
            block_origin, block_end = get_random_block_coordinates(inputs, self.inpaint_min_ratio,
                                                                   self.inpaint_max_ratio, origin_margin=3,
                                                                   spatial_dims=spatial_dims)
            block_size = block_end - block_origin
            block_values = torch.rand(block_size.tolist())

            outputs = replace_block(outputs, block_values, block_origin, block_end, spatial_dims)
            i += 1
        return outputs

    def get_outpainted(self, inputs: torch.Tensor) -> torch.Tensor:
        channels, *spatial_shape = inputs.shape
        spatial_dims = len(spatial_shape)
        slice_dims = range(-spatial_dims, 0)

        i = 0
        stop_painting = False
        noise = torch.rand(*spatial_shape).unsqueeze(0)
        noise = noise.repeat(channels, *([1] * len(spatial_shape)))
        outputs = noise

        while (i < self.outpaint_max_count) and not stop_painting:
            block_origin, block_end = get_random_block_coordinates(inputs, self.inpaint_min_ratio,
                                                                   self.inpaint_max_ratio, origin_margin=3,
                                                                   spatial_dims=spatial_dims)
            block_values = slice_select(inputs, block_origin, block_end, slice_dims)
            outputs = replace_block(outputs, block_values, block_origin, block_end, spatial_dims)
            i += 1
        return outputs


class GenesisPreprocessor(nn.Module):
    def __init__(self):
        super(GenesisPreprocessor, self).__init__()
        self.random_flip = RandomNDFlip()
        self.random_local_shuffle = RandomNDLocalShuffle()
        self.random_non_linear_transform = RandomNonLinearTransformation()
        self.random_painting = RandomInOutPainting(in_rate=0.5, outpaint_max_count=10)

    def forward(self, inputs):
        with torch.no_grad():
            outputs, inputs = self.apply_transforms(inputs)
        return outputs, inputs

    def apply_transforms(self, inputs):
        inputs = self.random_flip(inputs)
        outputs = self.random_local_shuffle(inputs.clone())
        outputs = self.random_non_linear_transform(outputs)
        outputs = self.random_painting(outputs)
        return outputs, inputs
