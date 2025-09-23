import torch
from torch.nn.functional import interpolate
import numpy as np
import cv2
from pathlib import Path

from data import ModalitySet, Modality, ModalityType
from analysis.visualization.visualizer import Visualizer
from analysis.visualization.attention_recorder import AttentionRecorder, AttentionModule
from models.module import MindfulModule
from models.model_output import ClassifierOutput
from models.attention_interface import AttentionInterface
from models.representation.encoders import ViTEncoder, ViT


class AttentionMap(Visualizer):
    visualizer_name = "attention_maps"

    def __init__(self,
                 model: MindfulModule,
                 log_dir: str | Path,
                 saved_slices: str | list[str] | tuple[str, ...] = ("center", "max_intensity", "depth_wise_max"),
                 saved_versions: str | list[str] | tuple[str, ...] = ("next_to_overlay",
                                                                      "next_to_additive",
                                                                      "next_to_mul"),
                 color_map: str | int = cv2.COLORMAP_JET,
                 mask_threshold: float = 0.25,
                 overlay_coeff=0.1,
                 resize_method=cv2.INTER_LINEAR,
                 input_modalities: ModalitySet = None,
                 **kwargs
                 ):
        super(AttentionMap, self).__init__(model=model,
                                           log_dir=log_dir,
                                           saved_slices=saved_slices,
                                           saved_versions=saved_versions,
                                           color_map=color_map,
                                           mask_threshold=mask_threshold,
                                           overlay_coeff=overlay_coeff,
                                           resize_method=resize_method,
                                           input_modalities=input_modalities
                                           )
        self.recorder = AttentionRecorder(model)
        self.module_names = {module: module_name for module_name, module in model.named_modules()
                             if module in self.recorder.attention_modules}
        self._kwargs = kwargs

    def __call__(self,
                 sample: list[torch.Tensor],
                 **kwargs
                 ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            batched_sample = [modality.unsqueeze(0) for modality in sample]

            attention_maps: dict[AttentionModule, torch.Tensor]
            _, attention_maps, module_inputs = self.recorder(batched_sample)

        outputs: dict[str, torch.Tensor] = {}
        for i, (attention_module, module_maps) in enumerate(attention_maps.items()):
            module_name = self.module_names[attention_module]

            if self.is_image_module(attention_module):
                module_maps = self.format_visual_attention_maps(module_maps, attention_module)
                outputs[module_name] = module_maps[0]

            elif self.is_1d_attention_module(attention_module):
                module_maps = attention_module.format_1d_attention(module_maps)
                outputs[module_name] = module_maps[0]

        return outputs

    def run_batch(self, batch: list[torch.Tensor], ids: list[str] = None) -> None:
        inputs, labels, _ = self.prepare_batch(batch)

        batch_size = self.get_batch_size(inputs)
        ids = self.get_ids(batch_size, ids)

        if isinstance(inputs, (list, tuple)) and len(inputs) == 1:
            inputs = inputs[0]

        attention_maps: dict[AttentionModule, torch.Tensor]
        predictions: ClassifierOutput
        predictions, attention_maps, modules_inputs = self.recorder(inputs)
        # visual attention_maps shape: [batch_size, n_layers, n_heads, patch_count + 1, patch_count + 1]
        # fusion attention_maps shape: [batch_size, n_layers, n_modalities, n_modalities]
        # predictions.logits: [batch_size]

        for i, (attention_module, module_maps) in enumerate(attention_maps.items()):
            i: int
            # Images (3D expected)
            if self.is_image_module(attention_module):
                module_maps = self.format_visual_attention_maps(module_maps, attention_module)

                fake_modality = Modality(ModalityType.IMAGE, modality_id="ViT_{}".format(i))
                self.aggregate(fake_modality, module_maps, ids)

                correct_predictions = self.get_correct_predictions(predictions, labels)
                self.save_images_3d(module_maps, ids, correct_predictions)

                vit_inputs = inputs[i] if isinstance(inputs, (list, tuple)) else inputs
                self.save_visual_attention_maps(vit_inputs, module_maps, correct_predictions, ids)

            # Non-image
            elif self.is_1d_attention_module(attention_module):
                formatted_maps = attention_module.format_1d_attention(module_maps)
                self.save_1d_attention(attention_module, formatted_maps, ids, method="Rollout")

                module_inputs = modules_inputs[attention_module]
                if module_inputs is not None:
                    inputs_norm = torch.norm(module_inputs, dim=-1)
                    self.save_1d_attention(attention_module, inputs_norm[:, 0], ids, method="InputsNorm")

                    weighted_attention = inputs_norm.unsqueeze(-1).unsqueeze(-1) * module_maps
                    weighted_attention = attention_module.format_1d_attention(weighted_attention)
                    self.save_1d_attention(attention_module, weighted_attention, ids, method="WeightedAttention")

    # region Visual attention maps (from ViTs)
    def save_visual_attention_maps(self,
                                   vit_inputs: torch.Tensor,
                                   attention_maps: torch.Tensor,
                                   correct_predictions: list[bool | None],
                                   ids: list
                                   ) -> None:
        vit_inputs = torch.movedim(vit_inputs, 1, -1).cpu().numpy()
        attention_maps = torch.unsqueeze(attention_maps, dim=-1).cpu().numpy()
        for base_input, attention_map, sample_id, correct_prediction \
                in zip(vit_inputs, attention_maps, ids, correct_predictions):
            self._save_output_slices(attention_map, "RolloutAttention",
                                     sample_id, correct_prediction, base_input)

    def format_visual_attention_maps(self,
                                     attention_maps: torch.Tensor,
                                     attention_module: ViT | ViTEncoder
                                     ) -> torch.Tensor:
        attention_maps, _ = attention_maps.min(dim=2)
        attention_maps = AttentionInterface.rollout_attention_over_layers(attention_maps)

        # attention_maps: [batch_size, n_heads, patch_count + 1, patch_count + 1]
        attention_maps = self.reduce_visual_pooling_attention(attention_module, attention_maps)
        # attention_maps: [batch_size, n_heads, patch_count]
        attention_maps = self.reshape_visual_attention_maps(attention_module, attention_maps)
        # attention_maps: [batch_size, *image_size]

        return attention_maps

    @staticmethod
    def reduce_visual_pooling_attention(module: ViT | ViTEncoder, attention_maps: torch.Tensor):
        pooling_method = None
        if isinstance(module, ViTEncoder):
            pooling_method = module.pooling.pooling

        elif isinstance(module, ViT):
            if hasattr(module, "classification_head"):
                pooling_method = "cls" if hasattr(module, "cls_token") else "first"
            else:
                pooling_method = "none"

        if pooling_method in [0, 1]:
            pooling_method = "first"
        elif pooling_method is None:
            pooling_method = "none"

        if pooling_method == "cls":
            attention_maps = attention_maps[:, 0, 1:]
        elif pooling_method == "first":
            attention_maps = attention_maps[:, 0]
        else:
            raise NotImplementedError("Unsupported pooling method: `{}`".format(pooling_method))

        return attention_maps

    def reshape_visual_attention_maps(self,
                                      vit: ViT,
                                      attention_maps: torch.Tensor
                                      ) -> torch.Tensor:
        attention_maps_shape = attention_maps.shape
        if attention_maps_shape[-1] != vit.total_patch_count:
            raise ValueError("Expected last dimension of `attention_maps` to be {}, got {}.".
                             format(vit.total_patch_count, attention_maps_shape[-1]))

        map_count = np.prod(attention_maps_shape[:-1])
        attention_maps = torch.reshape(attention_maps, (map_count, 1) + vit.patch_count)
        resize_method = "nearest" if self.resize_method == cv2.INTER_NEAREST else "trilinear"
        attention_maps = interpolate(attention_maps, size=vit.img_size, mode=resize_method)
        attention_maps = torch.reshape(attention_maps, attention_maps_shape[:-1] + vit.img_size)
        return attention_maps

    # endregion

    def save_1d_attention(self,
                          attention_module: AttentionInterface,
                          attention_maps: torch.Tensor,
                          ids: list,
                          method: str
                          ) -> None:
        attention_maps = attention_maps.cpu().numpy()

        module_name = self.module_names[attention_module]
        output_file: Path = self.visualizer_folder / "{}_{}_attention.npz".format(module_name, method)
        if output_file.exists():
            existing_data = np.load(output_file.as_posix(), allow_pickle=True)
            attention_maps = np.concatenate([existing_data["attention_maps"], attention_maps], axis=0)
            ids = np.concatenate([existing_data["ids"], ids], axis=0)

        self.visualizer_folder.mkdir(parents=True, exist_ok=True)
        np.savez(output_file, attention_maps=attention_maps, ids=ids)

    @staticmethod
    def is_image_module(attention_module: AttentionModule) -> bool:
        return isinstance(attention_module, ViT)

    @staticmethod
    def is_1d_attention_module(attention_module: AttentionModule) -> bool:
        if isinstance(attention_module, AttentionInterface):
            return attention_module.attention_rank == 1
        else:
            return False
