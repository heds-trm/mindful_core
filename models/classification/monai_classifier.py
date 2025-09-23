import torch.nn as nn
from monai.networks.nets.resnet import resnet50, resnet101, resnet152, resnet200
from monai.networks.nets.densenet import DenseNet121
from monai.networks.nets.vit import ViT
from monai.networks.nets.efficientnet import EfficientNetBN
from monai.networks.nets.senet import SEResNet50
from typing import Any

from models.model_output import ClassifierOutput
from models.classification.abstract_classifier import AbstractClassifier

SUPPORTED_BACKBONES = ["resnet", "resnet50", "resnet101", "resnet152", "resnet200",
                       "vit",
                       "densenet", "densenet121",
                       "efficientnet", "efficientnet-b0",
                       "seresnet", "seresnet50"]


class MonaiClassifier(AbstractClassifier):
    """
        todo: write documentation
    """

    # region Class/Subclass methods
    @classmethod
    def module_identifier(cls) -> str:
        return "monai_classifier"

    @classmethod
    def module_aliases(cls) -> tuple[str, ...]:
        return tuple(SUPPORTED_BACKBONES)

    @classmethod
    def multiview(cls) -> bool | int:
        return False

    @classmethod
    def required_sub_modules(cls) -> list[str]:
        return []

    # endregion
    def __init__(self,
                 model_name: str,
                 in_channels: int = 1,
                 spatial_dims: int = 3,
                 feed_forward: bool = True,
                 class_count: int = 1,
                 yield_confidence: bool = False,
                 label_smoothing=0.0,
                 backbone_config: dict[str, Any] = None,
                 optimizer_config: dict[str, dict[str, Any]] = None,
                 **kwargs
                 ):
        super(MonaiClassifier, self).__init__(class_count=class_count,
                                              optimizer_config=optimizer_config,
                                              label_smoothing=label_smoothing,
                                              single_class=class_count == 1,
                                              **kwargs)
        self.model_name = model_name
        self.feed_forward = feed_forward
        self._yield_confidence = yield_confidence
        self.backbone_config = backbone_config or {}

        output_size = class_count
        if feed_forward and self.yield_confidence:
            output_size += 1

        self.backbone = make_monai_classifier(model_name=model_name,
                                              in_channels=in_channels,
                                              output_size=output_size,
                                              spatial_dims=spatial_dims,
                                              feed_forward=feed_forward,
                                              **self.backbone_config)

    def forward(self, inputs, *args, **kwargs) -> ClassifierOutput:
        outputs = self.backbone(inputs)

        if self.yield_confidence and self.feed_forward:
            logits, confidence = outputs[..., :-1], outputs[..., -1]
        else:
            logits, confidence = outputs, None

        return ClassifierOutput(single_class=self.single_class,
                                logits=logits,
                                confidence=confidence,
                                confidence_threshold=self.confidence_threshold)

    @property
    def yield_confidence(self) -> bool:
        return self._yield_confidence

    def get_target_layer_name(self) -> str:
        if "resnet" in self.model_name:
            target_layer_name = "layer4"
        elif "densenet" in self.model_name:
            target_layer_name = "class_layers.relu"
        else:
            raise RuntimeError("`{}` is either currently not implemented or unsupported (requires CNNs)."
                               .format(self.model_name))
        return "backbone." + target_layer_name

    def get_fully_connected_layer_name(self) -> str:
        if "resnet" in self.model_name:
            fc_layer_name = "fc"
        elif "densenet" in self.model_name:
            fc_layer_name = "class_layers.out"
        else:
            raise RuntimeError("`{}` is either currently not implemented or unsupported (requires CNNs)."
                               .format(self.model_name))
        return "backbone." + fc_layer_name


def make_monai_classifier(model_name: str,
                          in_channels: int,
                          output_size: int,
                          spatial_dims: int = 3,
                          feed_forward: bool = True,
                          **kwargs) -> nn.Module:
    if model_name in ["densenet", "densenet121"]:
        return DenseNet121(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=output_size, **kwargs)

    elif model_name in ["resnet", "resnet50"]:
        return resnet50(spatial_dims=spatial_dims, n_input_channels=in_channels, num_classes=output_size,
                        feed_forward=feed_forward, **kwargs)
    elif model_name == "resnet101":
        return resnet101(spatial_dims=spatial_dims, n_input_channels=in_channels, num_classes=output_size,
                         feed_forward=feed_forward, **kwargs)
    elif model_name == "resnet152":
        return resnet152(spatial_dims=spatial_dims, n_input_channels=in_channels, num_classes=output_size,
                         feed_forward=feed_forward, **kwargs)
    elif model_name == "resnet200":
        return resnet200(spatial_dims=spatial_dims, n_input_channels=in_channels, num_classes=output_size,
                         feed_forward=feed_forward, **kwargs)

    elif model_name == "vit":
        return ViT(spatial_dims=spatial_dims, in_channels=in_channels, num_classes=output_size,
                   classification=feed_forward, **kwargs)

    elif model_name == "efficientnet-b0":
        return EfficientNetBN("efficientnet-b0", spatial_dims=spatial_dims, num_classes=output_size,
                              in_channels=in_channels, **kwargs)

    elif model_name == "seresnet50":
        return SEResNet50(spatial_dims=spatial_dims, num_classes=output_size, in_channels=in_channels, **kwargs)

    else:
        raise NotImplementedError(model_name)
