import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from monai.networks.nets import ViT
from monai.networks.blocks import SABlock
from typing import Callable, Any

from models.attention_interface import AttentionInterface

# from models.representation.encoders.multimodal import FusionTransformer, FusionTransformerEncoderLayer

AttentionModule = ViT | AttentionInterface
ModuleRecordings = torch.Tensor | dict[str, torch.Tensor]

AttentionModuleData = dict[AttentionModule, list[torch.Tensor] | torch.Tensor | None]
AttentionHook = Callable[[nn.Module, Any, Any], None]


class AttentionRecorder(nn.Module):
    def __init__(self, model: ViT | nn.Module, target_device: str = None):
        super(AttentionRecorder, self).__init__()

        attention_modules = self.find_attention_modules(model)
        if len(attention_modules) == 0:
            raise ValueError("Could not find any attention module in `model`.")

        self.model = model
        self.attention_modules: list[AttentionModule] = attention_modules
        self.attention_layers: dict[nn.Module, AttentionModule] = {}

        self.data = None
        self.recordings: dict[nn.Module, ModuleRecordings | list[ModuleRecordings]] = {}
        self.hooks: list[RemovableHandle] = []
        self.hook_registered = False
        self.ejected = False
        self.target_device = target_device

    @staticmethod
    def find_attention_modules(model: nn.Module) -> list[AttentionModule]:
        if isinstance(model, AttentionModule):
            attention_modules = model.get_attention_modules()
            if not isinstance(attention_modules, list):
                attention_modules = [attention_modules]
            return attention_modules

        sub_modules = [module for module in model.modules() if module != model]
        attention_modules = [module for module in sub_modules if
                             isinstance(module, AttentionModule)]

        if len(attention_modules) >= 1:
            return attention_modules
        sub_modules = sum([AttentionRecorder.find_attention_modules(sub_module) for sub_module in sub_modules], [])
        return sub_modules

    # region Hooks
    def _register_attention_hooks(self):
        for attention_module in self.attention_modules:
            if isinstance(attention_module, ViT):
                attention_layers = [layer.drop_weights
                                    for layer in attention_module.blocks.modules()
                                    if isinstance(layer, SABlock)]
                hook = self._vit_dropout_hook

            elif isinstance(attention_module, AttentionInterface):
                attention_layers = attention_module.get_attention_layers()
                hook = self._attention_interface_hook

            else:
                raise TypeError(type(attention_module))

            for attention_layer in attention_layers:
                self._register_attention_layer_forward_hook(attention_layer, attention_module, hook)

        self.hook_registered = True

    def _register_attention_layer_forward_hook(self,
                                               attention_layer: nn.Module,
                                               attention_module: AttentionModule,
                                               hook: AttentionHook
                                               ) -> None:
        handle = attention_layer.register_forward_hook(hook)
        self.hooks.append(handle)
        self.attention_layers[attention_layer] = attention_module

    def _vit_dropout_hook(self, layer, inputs: tuple[torch.Tensor, ...], __) -> None:
        self.add_recordings(layer, inputs[0])

    def _attention_interface_hook(self, layer, inputs: Any, outputs: Any) -> None:
        attention_interface: AttentionInterface = self.attention_layers[layer]
        recordings = attention_interface.get_attention_recordings(inputs, outputs)
        if not isinstance(recordings, (list, tuple)):
            recordings = [recordings]

        self.add_recordings(layer, *recordings)

    def eject(self) -> list[AttentionModule]:
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.attention_modules

    # endregion

    # region Recordings
    @staticmethod
    def detach_recordings(recordings: ModuleRecordings) -> ModuleRecordings:
        if isinstance(recordings, torch.Tensor):
            recordings = recordings.clone().detach()
        else:
            recordings = {key: tensor.clone().detach() for key, tensor in recordings.items()}
        return recordings

    def add_recordings(self, layer: nn.Module, *recordings: ModuleRecordings) -> None:
        recordings = [self.detach_recordings(recording) for recording in recordings]
        if len(recordings) == 1:
            recordings = recordings[0]
        self.recordings[layer] = recordings

    def clear(self):
        self.recordings.clear()

    # endregion

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, AttentionModuleData, AttentionModuleData]:
        # region Initialization
        if self.ejected:
            raise RuntimeError("This recorder has been ejected and cannot be used anymore.")
        self.clear()

        if not self.hook_registered:
            self._register_attention_hooks()

        ref_tensor = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        target_device = self.target_device if self.target_device else ref_tensor.device
        # endregion

        predictions = self.model(inputs)

        if len(self.recordings) == 0:
            return predictions, {}, {}

        attention_maps: AttentionModuleData = {module: [] for module in self.attention_modules}
        modules_inputs: AttentionModuleData = {module: [] for module in self.attention_modules}
        for layer, recording in self.recordings.items():
            if isinstance(recording, list):
                attention_weights, module_inputs = recording
            else:
                attention_weights, module_inputs = recording, None

            attention_module = self.attention_layers[layer]
            attention_weights = attention_weights.to(target_device)
            attention_maps[attention_module].append(attention_weights)

            if module_inputs is not None:
                module_inputs = module_inputs.to(target_device)
                modules_inputs[attention_module].append(module_inputs)

        for attention_module in self.attention_modules:
            module_maps = attention_maps[attention_module]
            if len(module_maps) > 0:
                attention_maps[attention_module] = torch.stack(attention_maps[attention_module], dim=1)
            else:
                attention_maps[attention_module] = None

            module_inputs = modules_inputs[attention_module]
            if len(module_inputs) > 0:
                modules_inputs[attention_module] = torch.stack(modules_inputs[attention_module], dim=1)
            else:
                modules_inputs[attention_module] = None

        # visual attention_maps shape: [batch_size, n_layers, n_heads, patch_count + 1, patch_count + 1]
        # fusion attention_maps shape: [batch_size, n_layers, n_heads, n_modalities, n_modalities]
        return predictions, attention_maps, modules_inputs
