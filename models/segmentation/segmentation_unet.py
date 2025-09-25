import torch
from monai.networks.nets import UNet, UNETR, SwinUNETR
from monai.losses.dice import DiceCELoss, DiceFocalLoss
from typing import Any, Sequence

from mindful_core.models.model_output import SegmentationOutput
from mindful_core.models.segmentation.abstract_segmentation_model import AbstractSegmentationModel

UNET_HPARAMS = ["kernel_size", "up_kernel_size", "num_res_units", "act", "norm", "dropout", "bias", "adn_ordering"]
UNETR_HPARAMS = ["feature_size", "hidden_size", "mlp_dim", "num_heads", "pos_embed", "proj_type",
                 "norm_name", "conv_block", "res_block", "dropout_rate", "qkv_bias", "save_attn"]
SWIN_HPARAMS = ["depths", "num_heads", "feature_size", "norm_name", "drop_rate", "attn_drop_rate", "dropout_path_rate",
                "normalize", "use_checkpoint", "downsample", "use_v2"]


class SegmentationUNet(AbstractSegmentationModel):

    # region Class/Subclass methods
    @classmethod
    def module_identifier(cls) -> str:
        return "segmentation_unet"

    # endregion
    def __init__(self,
                 class_count: int,
                 in_channels=1,
                 channels=(16, 32, 64, 128, 256),
                 strides=(2, 2, 2, 2),
                 optimizer_config: dict[str, dict[str, Any]] = None,
                 use_focal_loss: bool = False,
                 backbone: str | None = None,
                 img_size: Sequence[int] | int | None = None,
                 use_transformer: bool | None = None,
                 roi_size: Sequence[int] | None = None,
                 **kwargs):
        super().__init__(class_count=class_count,
                         optimizer_config=optimizer_config,
                         roi_size=roi_size)

        self.in_channels = in_channels
        self.channels = channels
        self.strides = strides
        self.use_focal_loss = use_focal_loss
        self.use_transformer = use_transformer
        self.img_size = img_size

        roi_size = img_size if roi_size is None else roi_size
        self.roi_size = torch.Size(roi_size) if roi_size is not None else None

        # Backwards compatibility
        if (backbone is None) and (use_transformer is not None):
            backbone = "unetr" if use_transformer else "unet"
        backbone = "unet" if backbone is None else backbone.lower()
        backbone_hparams = {"in_channels": in_channels,
                            "out_channels": class_count,
                            "spatial_dims": 3,
                            }
        img_size_required = backbone in ["unetr", "swin"]
        if img_size_required and (img_size is None):
            raise ValueError("`img_size` must be provided when using a transformer-based UNet.")

        if backbone == "unet":
            unet_kwargs = {key: value for key, value in kwargs.items() if key in UNET_HPARAMS}
            self.backbone = UNet(channels=channels, strides=strides, **backbone_hparams, **unet_kwargs)
        elif backbone == "unetr":
            unetr_kwargs = {key: value for key, value in kwargs.items() if key in UNETR_HPARAMS}
            self.backbone = UNETR(img_size=img_size, **backbone_hparams, **unetr_kwargs)
        elif backbone == "swin":
            swin_kwargs = {key: value for key, value in kwargs.items() if key in SWIN_HPARAMS}
            self.backbone = SwinUNETR(img_size=img_size, **backbone_hparams, **swin_kwargs)
        else:
            raise ValueError(backbone)

        if use_focal_loss:
            loss_module = DiceFocalLoss(to_onehot_y=True, softmax=True)
        else:
            loss_module = DiceCELoss(to_onehot_y=True, softmax=True)
        self._loss_module = loss_module

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> SegmentationOutput:
        logits = self.backbone(inputs)
        return SegmentationOutput(single_class=self.single_class, logits=logits)


