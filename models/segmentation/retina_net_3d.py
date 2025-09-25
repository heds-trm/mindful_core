import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any, Literal, Sequence

from mindful_core.data.subset_id import SubsetID
from mindful_core.models.module import MindfulModule
from mindful_core.models.model_output import BoundingBoxOutput
from mindful_core.models.loss_aggregator import LossAggregator


# region ResNet blocks
class AbstractResNetBlock(nn.Module):
    expansion = 0

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 allow_downsample: bool = False,
                 ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.allow_downsample = allow_downsample

        if self.perform_downsample:
            self.conv_downsample = nn.Conv3d(in_channels, self.expanded_out_channels, kernel_size=1, stride=stride,
                                             bias=False)
            self.batch_norm_downsample = nn.BatchNorm3d(self.expanded_out_channels)
        else:
            self.conv_downsample = None
            self.batch_norm_downsample = None

    @property
    def perform_downsample(self) -> bool:
        return ((self.stride != 1) or (self.in_channels != self.expanded_out_channels)) and self.allow_downsample

    @property
    def expanded_out_channels(self) -> int:
        return self.out_channels * self.expansion


class ResNetBlock(AbstractResNetBlock):
    expansion = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 allow_downsample: bool = False,
                 ) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         stride=stride,
                         allow_downsample=allow_downsample,
                         )

        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, stride=stride)
        self.batch_norm_1 = nn.BatchNorm3d(out_channels)

        self.conv_2 = nn.Conv3d(out_channels, self.expanded_out_channels, kernel_size=3, padding=1, bias=False,
                                stride=1)
        self.batch_norm_2 = nn.BatchNorm3d(self.expanded_out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.batch_norm_2(x)

        if self.perform_downsample:
            residual = self.conv_downsample(residual)
            residual = self.batch_norm_downsample(residual)

        x += residual
        x = self.relu(x)

        return x


class ResNetBottleneck(AbstractResNetBlock):
    expansion = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 allow_downsample: bool = False,
                 ) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         stride=stride,
                         allow_downsample=allow_downsample,
                         )

        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm3d(out_channels)

        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, stride=stride)
        self.batch_norm_2 = nn.BatchNorm3d(out_channels)

        self.conv_3 = nn.Conv3d(out_channels, self.expanded_out_channels, kernel_size=1, bias=False)
        self.batch_norm_3 = nn.BatchNorm3d(self.expanded_out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)

        x = self.conv_3(x)
        x = self.batch_norm_3(x)

        if self.perform_downsample:
            residual = self.conv_downsample(residual)
            residual = self.batch_norm_downsample(residual)

        x += residual
        x = self.relu(x)

        return x


# endregion

# region FPN (Feature Pyramid Network)
class PyramidBlock(nn.Module):
    def __init__(self, in_channels: int, feature_size: int, is_first: bool, is_last: bool) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.feature_size = feature_size
        self.is_first = is_first
        self.is_last = is_last

        self.conv_1 = nn.Conv3d(in_channels, feature_size, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv3d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        if is_first:
            self.upsample = None
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor | None]
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x, x_upsampled = inputs

        x = self.conv_1(x)

        if not self.is_last:
            x = x + x_upsampled

        if not self.is_first:
            x_upsampled = self.upsample(x)

        x = self.conv_2(x)

        if self.is_first:
            return x, None
        else:
            return x, x_upsampled


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, feature_size: int, *input_sizes: int, total_block_count: int = 5) -> None:
        super().__init__()

        self.feature_size = feature_size
        self.input_sizes = input_sizes
        self.total_block_count = total_block_count

        self.pyramid_blocks: list[PyramidBlock] = []
        for i, input_size in enumerate(input_sizes):
            is_first = i == 0
            is_last = i == (len(input_sizes) - 1)
            pyramid_block = PyramidBlock(input_size, self.feature_size, is_first, is_last)
            self.pyramid_blocks.append(pyramid_block)

        self.extra_layers = []
        extra_layer_count = total_block_count - len(self.pyramid_blocks)
        input_size = input_sizes[-1]
        for i in range(extra_layer_count):
            layer = nn.Conv3d(input_size, feature_size, kernel_size=3, stride=2, padding=1)
            input_size = feature_size
            if i > 0:
                layer = nn.Sequential(nn.ReLU(), layer)
            self.extra_layers.append(layer)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(inputs) != len(self.pyramid_blocks):
            raise ValueError(len(inputs), len(self.pyramid_blocks))

        outputs: list[torch.Tensor] = []
        x_upsampled = None
        for i, pyramid_block in reversed(list(enumerate(self.pyramid_blocks))):
            x = inputs[i]
            x, x_upsampled = pyramid_block((x, x_upsampled))
            outputs.insert(0, x)

        x = inputs[-1]
        for additional_layer in self.extra_layers:
            x = additional_layer(x)
            outputs.append(x)

        return outputs


# endregion

# region Bounding box regression / Classification
class RetinaRegressor(nn.Module):
    def __init__(self,
                 in_channels: int,
                 channels_per_anchor: int,
                 anchors_count: int,
                 feature_size: int = 256,
                 depth: int = 5
                 ) -> None:
        super().__init__()

        self.input_size = in_channels
        self.channels_per_anchor = channels_per_anchor
        self.anchors_count = anchors_count
        self.feature_size = feature_size
        self.depth = depth

        self.layers = []
        for i in range(depth):
            layer_in_channels = in_channels if i == 0 else feature_size
            layer_out_channels = self.output_size if i == (depth - 1) else feature_size
            layer = nn.Conv3d(layer_in_channels, layer_out_channels, kernel_size=3, padding=1)
            self.layers.append(layer)
            if i != (depth - 1):
                self.layers.append(nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.anchors_count, self.channels_per_anchor)
        return x

    @property
    def output_size(self) -> int:
        return self.channels_per_anchor * self.anchors_count

    @property
    def output_layer(self) -> nn.Module:
        return self.layers[-1]


# endregion

# region Focal Loss
def compute_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    iou = intersection / ua

    return iou


class FocalLoss(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().cuda())

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue

            iou = compute_iou(anchors[0, :, :], bbox_annotation[:, :4])  # num_anchors x num_annotations

            iou_max, iou_argmax = torch.max(iou, dim=1)  # num_anchors x 1

            # import pdb
            # pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(iou_max, 0.4), :] = 0

            positive_indices = torch.ge(iou_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[iou_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                # negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        classification_loss = torch.stack(classification_losses).mean(dim=0, keepdim=True)
        regression_loss = torch.stack(regression_losses).mean(dim=0, keepdim=True)
        return classification_loss, regression_loss


# endregion

class RetinaNet3D(MindfulModule):
    # region Class/Subclass methods
    @classmethod
    def module_identifier(cls) -> str:
        return "retina_net_3d"

    @classmethod
    def module_aliases(cls) -> tuple[str, ...]:
        return ("retina_net", )

    # endregion

    def __init__(self,
                 in_channels: int,
                 class_count: int,
                 hidden_size: int = 64,
                 resnet_block_mode: Literal["basic_block", "bottleneck"] = "bottleneck",
                 resnet_block_depth: Sequence[int] = (3, 4, 6, 3),
                 resnet_to_pyramid_count: int = 3,
                 pyramid_hidden_size: int = 256,
                 pyramid_total_block_count: int = 5,
                 optimizer_config: dict[str, dict[str, Any]] = None,
                 ):
        super().__init__(optimizer_config=optimizer_config)
        self.in_channels = in_channels
        self.class_count = class_count
        self.hidden_size = hidden_size

        self.resnet_block_mode = resnet_block_mode
        self.resnet_block_depth = resnet_block_depth
        self.resnet_to_pyramid_count = resnet_to_pyramid_count
        self.pyramid_hidden_size = pyramid_hidden_size
        self.pyramid_total_block_count = pyramid_total_block_count

        self.stem = self._make_stem()
        self.resnet_output_sizes = []
        self.resnet_layers = self._make_resnet_layers()
        self.feature_pyramid_network = self._make_feature_pyramid_network()
        self.bounding_box_regressor = RetinaRegressor(pyramid_hidden_size, channels_per_anchor=4, anchors_count=9)
        self.classifier = RetinaRegressor(pyramid_hidden_size, channels_per_anchor=class_count, anchors_count=9)

        # self.anchors = Anchors()
        # self.regression_boxes = BBoxTransform()
        # self.clip_boxes = ClipBoxes()
        self.focal_loss = FocalLoss()

        self._init_layer_weights()
        self.freeze_bn()

    # region Initialization
    def _make_stem(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(self.in_channels, self.hidden_size, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

    def _make_resnet_layers(self) -> list[nn.Sequential]:
        block_class = ResNetBottleneck if self.resnet_block_mode == "bottleneck" else ResNetBlock

        layers = []
        in_channels = self.hidden_size
        for i in range(self.resnet_block_count):
            stride = 2 if i > 0 else 1
            out_channels = self.hidden_size * (2 ** i)
            layer = self._make_resnet_layer(in_channels, out_channels, stride, self.resnet_block_depth[i])
            layers.append(layer)
            in_channels = out_channels * block_class.expansion
            self.resnet_output_sizes.append(in_channels)

        return layers

    def _make_resnet_layer(self,
                           in_channels: int,
                           out_channels: int,
                           stride: int,
                           depth: int
                           ) -> nn.Sequential:
        block_class = ResNetBottleneck if self.resnet_block_mode == "bottleneck" else ResNetBlock
        blocks = [block_class(in_channels, out_channels, stride, allow_downsample=True)]

        in_channels = out_channels * block_class.expansion
        for i in range(depth):
            blocks.append(block_class(in_channels, out_channels, allow_downsample=False))

        return nn.Sequential(*blocks)

    def _make_feature_pyramid_network(self) -> FeaturePyramidNetwork:
        input_sizes = self.resnet_output_sizes[-self.resnet_to_pyramid_count:]
        return FeaturePyramidNetwork(self.pyramid_hidden_size, *input_sizes,
                                     total_block_count=self.pyramid_total_block_count)

    def _init_layer_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                # noinspection PyTypeChecker
                module.weight.data.normal_(0, torch.sqrt(2. / n))
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

        prior = 0.01
        self.classifier.output_layer.weight.data.fill_(0)
        # noinspection PyTypeChecker
        self.classifier.output_layer.bias.data.fill_(-torch.log((1.0 - prior) / prior))

        self.bounding_box_regressor.output_layer.weight.data.fill_(0)
        self.bounding_box_regressor.output_layer.bias.data.fill_(0)

    def freeze_bn(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    # endregion

    # region Forward
    def forward(self, inputs: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        x = self.stem(inputs)

        pyramid_inputs = []
        for i, resnet_block in enumerate(self.resnet_layers):
            x = resnet_block(x)
            if i >= self.pyramid_input_index_start:
                pyramid_inputs.append(x)

        pyramid_features = self.feature_pyramid_network(pyramid_inputs)

        bounding_box_regression = self.run_regressor(self.bounding_box_regressor, pyramid_features)
        logits = self.run_regressor(self.classifier, pyramid_features)

    @staticmethod
    def run_regressor(regressor: RetinaRegressor, pyramid_features: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([regressor(feature) for feature in pyramid_features], dim=1)

    # endregion

    def base_step(self, batch, subset_id: SubsetID, **model_kwargs) -> STEP_OUTPUT:
        self.loss_aggregator.clear()
        inputs, ground_truth = self.unpack_batch(batch)

        outputs: BoundingBoxOutput = self(inputs, **model_kwargs)

        self.compute_output_losses(outputs, ground_truth, subset_id)

        self.log_step_losses(subset_id)
        loss, logged_losses = self.loss_aggregator.get_logged_losses()
        step_outputs = {
            **logged_losses,
            "ground_truth": ground_truth,
            **outputs.detached_outputs(),
        }
        self.step_outputs[subset_id].append(step_outputs)
        return loss

    def compute_output_losses(self,
                              outputs: BoundingBoxOutput,
                              labels: torch.Tensor,
                              subset_id: SubsetID
                              ) -> LossAggregator:
        pass

    def log_epoch_outputs(self, subset: SubsetID):
        pass

    # region Properties
    @property
    def resnet_block_count(self) -> int:
        return len(self.resnet_block_depth)

    @property
    def pyramid_input_index_start(self) -> int:
        return self.resnet_block_count - self.resnet_to_pyramid_count

    # endregion


def main():
    model = RetinaNet3D(
        in_channels=1,
        class_count=3,
        hidden_size=64,
        resnet_block_mode="bottleneck",
        resnet_block_depth=(3, 4, 6, 3),
        resnet_to_pyramid_count=3,
        pyramid_hidden_size=256,
        pyramid_total_block_count=5
    )

    x = torch.randn(4, 1, 64, 64, 64)
    _ = model(x)


if __name__ == "__main__":
    main()
