import torch
from monai.transforms import SaveImage
from monai.data import MetaTensor
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from abc import abstractmethod, ABC
from typing import Literal, Sequence

from mindful_core.data import ModalitySet, ModalityType, Modality
from mindful_core.models.module import MindfulModule, ModelInput
from mindful_core.models.model_output import ClassifierOutput
from mindful_core.models.classification.abstract_classifier import AbstractClassifier
from mindful_core.analysis.visualization.radar_chart import radar_from_data_frame
from mindful_core.utils.imaging import get_3d_image_max_intensity_slice_index, get_center_of_mass
from mindful_core.utils.visualization import format_image


class Visualizer(ABC):
    visualizer_name = None

    def __init__(self,
                 model: MindfulModule,
                 log_dir: str | Path,
                 saved_slices: str | list[str] | tuple[str] | None = "center",
                 saved_versions: str | list[str] | tuple[str] | None = ("overlay", "next_to_overlay", "hsv"),
                 color_map: str | int | None = cv2.COLORMAP_JET,
                 mask_threshold: float | None = 0.25,
                 overlay_coeff: float | None = 0.1,
                 resize_method=cv2.INTER_NEAREST,
                 input_modalities: ModalitySet = None,
                 features_names: dict[Modality, list[str]] = None,
                 ):
        self.model = model
        self.log_dir = Path(log_dir)
        self.input_modalities = input_modalities

        # region Saved slices/versions
        if saved_slices is not None:
            if not isinstance(saved_slices, (tuple, list)):
                saved_slices = (saved_slices,)
        self.saved_slices = saved_slices

        if saved_versions is not None:
            if not isinstance(saved_versions, (tuple, list)):
                saved_versions = (saved_versions,)
        self.saved_versions = saved_versions

        # endregion

        # region Heatmap color maps
        if color_map is not None:
            if color_map == "jet":
                color_map = cv2.COLORMAP_JET
            elif color_map == "viridis":
                color_map = cv2.COLORMAP_VIRIDIS
            elif isinstance(color_map, str):
                raise ValueError("Unknown color map: {}".format(color_map))
        self.color_map = color_map

        # endregion

        # region Additional output images parameters
        self.mask_threshold = mask_threshold
        self.overlay_coeff = overlay_coeff
        self.resize_method = resize_method

        # endregion

        # region Vector features
        self.features_names = features_names or {}
        self.features_buffer: dict[Modality, pd.DataFrame] = {}

        # endregion

        # region Accumulators
        self.samples_seen_count = 0

        self.averages: dict[Modality, np.ndarray] = {}
        self.averages_count: dict[Modality, int] = {}
        self.centers_of_mass: dict[Modality, dict[str, tuple[int, int, int]]] = {}

        # endregion

        self._image_saver: SaveImage | None = None

    @abstractmethod
    def __call__(self,
                 sample: Sequence[torch.Tensor],
                 **kwargs
                 ) -> dict[str, torch.Tensor]:
        """
        Runs the visualizer for a simple sample and return a dictionary of the different outputs of the visualizer.
        :param sample: A single, un-batched, sample.
        :param kwargs:
        :return: A dictionary of the different outputs of the visualizer, identified by a unique string ID.
            The nature of the ID depends on the visualizer.
        """
        raise NotImplementedError

    @staticmethod
    def sample_as_batch(sample: ModelInput) -> ModelInput:
        if isinstance(sample, torch.Tensor):
            return sample.unsqueeze(0)
        else:
            return [Visualizer.sample_as_batch(modality) for modality in sample]

    @abstractmethod
    def run_batch(self, batch: ModelInput, ids: list[str] = None) -> None:
        raise NotImplementedError

    def aggregate(self, modality: Modality, modality_value: torch.Tensor, ids: list[str]) -> None:
        modality_value = modality_value.cpu().numpy()
        modality_value = np.squeeze(modality_value)

        if modality.type == ModalityType.IMAGE:
            if modality not in self.centers_of_mass:
                self.centers_of_mass[modality] = {}

            for image, image_id in zip(modality_value, ids):
                center_of_mass = get_center_of_mass(image)
                # noinspection PyTypeChecker
                self.centers_of_mass[modality][image_id] = center_of_mass

            # average over depth (expecting batch, width, height, [depth])
            if len(modality_value.shape) > 3:
                modality_value = modality_value.mean(-1)

        if modality not in self.averages:
            self.averages[modality] = modality_value.mean(0)
            self.averages_count[modality] = 1

        else:
            self.averages[modality] += modality_value.mean(0)
            self.averages_count[modality] += 1

    def finalize(self) -> None:
        for modality, modality_average in self.averages.items():
            modality_average = modality_average / self.averages_count[modality]

            if modality.type == ModalityType.IMAGE:
                image = self._format_image(modality_average, self.color_map)
                self.write_image(image, correct_prediction=None,
                                 sample_id="Average", image_name=self.__class__.__name__, modality=modality)

                image_centers_of_mass = self.centers_of_mass[modality]
                axes = ["X", "Y", "Z", "T"]
                spatial_dims = len(next(iter(image_centers_of_mass.values())))
                columns = axes[:spatial_dims]
                dataframe = pd.DataFrame.from_dict(image_centers_of_mass, orient="index", columns=columns)

                center_of_mass_path = self.visualizer_folder / "{}_center_of_mass.csv".format(modality.id)
                dataframe.to_csv(center_of_mass_path)

            # else:
            #     self.save_features(modality, modality_average, ids, name="occlusion")

    @staticmethod
    def get_batch_size(batch: ModelInput) -> int:
        if isinstance(batch, torch.Tensor):
            batch_size = batch.shape[0]
        else:
            batch_size = batch[0].shape[0]
        return batch_size

    def prepare_batch(self, batch: ModelInput) -> tuple[ModelInput, torch.Tensor | None, ModalitySet | None]:
        if self.input_modalities is None:
            if isinstance(self.model, AbstractClassifier):
                *inputs, labels = batch
            else:
                inputs, labels = batch, None
            input_modalities = None

        elif ModalityType.LABEL in self.input_modalities:
            inputs = [value for modality, value in zip(self.input_modalities, batch)
                      if modality.type != ModalityType.LABEL]
            labels = batch[self.input_modalities.get_index(ModalityType.LABEL)]
            input_modalities = self.input_modalities - ModalityType.LABEL
        else:
            inputs, labels = batch, None
            input_modalities = self.input_modalities

        return inputs, labels, input_modalities

    # region Images (2D)
    def save_batch_output_slices(self,
                                 inputs: torch.Tensor | np.ndarray | list[np.ndarray],
                                 outputs: torch.Tensor | np.ndarray | list[np.ndarray],
                                 ids: list[str],
                                 correct_predictions: list[bool] | np.ndarray | None,
                                 name: str,
                                 modality: Modality | str
                                 ) -> None:
        if correct_predictions is None:
            correct_predictions = [None] * len(inputs)

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()

        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()

        for input_tensor, output_tensor, input_id, correct_prediction \
                in zip(inputs, outputs, ids, correct_predictions):
            self._save_output_slices(output_tensor, name, input_id, modality, correct_prediction, input_tensor)

    def _save_output_slices(self,
                            output: np.ndarray,
                            output_name: str,
                            sample_id: str,
                            modality: Modality,
                            correct_prediction: bool | None,
                            base_image: np.ndarray,
                            ):
        if "max_intensity" in self.saved_slices:
            max_intensity_slice_index = get_3d_image_max_intensity_slice_index(output, dim=-2)
            output_slice = output[..., max_intensity_slice_index, :]
            base_slice = base_image[..., max_intensity_slice_index, :]
            self._save_output_image(output_slice,
                                    output_name="{}_MaxIntensitySlice".format(output_name),
                                    sample_id=sample_id,
                                    modality=modality,
                                    correct_prediction=correct_prediction,
                                    base_image=base_slice)

        if "depth_wise_max" in self.saved_slices:
            center_index = (output.shape[-2] - 1) // 2
            output_slice = np.nanmax(output, axis=-2)
            base_slice = base_image[..., center_index, :]
            self._save_output_image(output_slice,
                                    output_name="{}_DepthWiseMax".format(output_name),
                                    sample_id=sample_id,
                                    modality=modality,
                                    correct_prediction=correct_prediction,
                                    base_image=base_slice)

        if "center" in self.saved_slices:
            center_index = (output.shape[-2] - 1) // 2
            output_slice = output[..., center_index, :]
            base_slice = base_image[..., center_index, :]
            self._save_output_image(output_slice,
                                    output_name="{}_Center".format(output_name),
                                    sample_id=sample_id,
                                    modality=modality,
                                    correct_prediction=correct_prediction,
                                    base_image=base_slice)

    def _format_image(self,
                      image: np.ndarray,
                      color_map=None,
                      normalize_image=True,
                      resize_method=None,
                      target_resolution: int | None = None,
                      rotate: bool = True
                      ) -> np.ndarray:
        resize_method = resize_method if resize_method is not None else self.resize_method
        target_resolution = target_resolution if target_resolution is not None else 512
        return format_image(image, color_map, normalize_image, target_resolution, resize_method, rotate)

    def _save_output_image(self,
                           output: np.ndarray,
                           output_name: str,
                           sample_id: str,
                           modality: Modality,
                           correct_prediction: bool = None,
                           base_image: np.ndarray = None,
                           ):
        base_output = output
        output = self._format_image(output, self.color_map)
        self.write_image(output, correct_prediction, sample_id, output_name, modality)

        if (base_image is not None) and (len(self.saved_versions) > 0):
            base_image = self._format_image(base_image)
            if base_image.ndim == 2:
                base_image = np.expand_dims(base_image, axis=-1)

            if base_image.shape[2] == 1:
                base_image_3c = np.repeat(base_image, repeats=3, axis=2)
            else:
                base_image_3c = base_image

            # region Overlay attention/cam/saliency/... on base image
            if self._is_saved("overlay"):
                mask = self._format_image(base_output)
                if self.mask_threshold is None:
                    # mask_threshold = np.percentile(mask, 75)
                    mask_threshold = (mask.max() * 0.25).astype(mask.dtype)
                else:
                    mask_threshold = self.mask_threshold

                # iso_th = [np.percentile(mask, i) for i in [75, 85, 95, 97, 99]]
                iso_th = [(mask.max() * i / 100).astype(mask.dtype) for i in [25, 50, 75, 90]]
                iso_masks = [np.float32(mask > th) for th in iso_th]
                iso_mask = np.zeros_like(iso_masks[0])
                for i in range(len(iso_th)):
                    iso_mask_blurred = cv2.blur(iso_masks[i], ksize=(4, 4))
                    iso_line = (iso_mask_blurred - iso_masks[i]) > 0
                    iso_mask = np.maximum(iso_mask, np.float32(iso_line) * iso_th[i])
                iso_lines = cv2.applyColorMap(np.uint8(iso_mask), self.color_map)
                iso_mask = np.expand_dims(np.float32(iso_mask > 0), axis=-1)

                mask = mask > mask_threshold
                mask = np.expand_dims(np.float32(mask), axis=-1)

                masked_output = np.float32(output)
                masked_output = masked_output * self.overlay_coeff + base_image * (1.0 - self.overlay_coeff)
                overlaid_image = base_image * (1.0 - mask) + masked_output * mask
                overlaid_image = overlaid_image * (1.0 - iso_mask) + iso_lines * iso_mask
                # overlaid_image = (base_image + masked_output * self.overlay_coeff) / (1.0 + self.overlay_coeff)

                if "overlay" in self.saved_versions:
                    self.write_image(overlaid_image, correct_prediction, sample_id,
                                     image_name="{}_Overlay".format(output_name), modality=modality)

                if "next_to_overlay" in self.saved_versions:
                    stacked_images = cv2.hconcat([base_image_3c.astype(overlaid_image.dtype), overlaid_image])
                    self.write_image(stacked_images, correct_prediction, sample_id,
                                     image_name="{}_NextToOverlay".format(output_name), modality=modality)
            # endregion

            # region Use attention/cam/saliency/... as value (HSV) for base image
            if self._is_saved("hsv"):
                base_hsv = self.apply_mask(base_image, base_output, "hue", isoline_thresholds=[0.25, 0.5, 0.75, 0.90])
                # base_hsv = cv2.cvtColor(base_image_3c.astype(np.float32), cv2.COLOR_BGR2HSV)
                # output_hsv = cv2.cvtColor(output.astype(np.float32), cv2.COLOR_BGR2HSV)
                # base_hsv[..., :2] = output_hsv[..., :2]
                # base_hsv = cv2.cvtColor(base_hsv, cv2.COLOR_HSV2BGR)

                if "hsv" in self.saved_versions:
                    self.write_image(base_hsv, correct_prediction, sample_id,
                                     image_name="{}_HSV".format(output_name), modality=modality)

                if "next_to_hsv" in self.saved_versions:
                    stacked_images = cv2.hconcat([base_image_3c.astype(base_hsv.dtype), base_hsv])
                    self.write_image(stacked_images, correct_prediction, sample_id,
                                     image_name="{}_NextToHSV".format(output_name), modality=modality)
            # endregion

            # region Multiply attention/cam/saliency/... with base image
            if self._is_saved("mul"):
                # factor = np.expand_dims(self._convert_image(base_output), axis=-1)
                # multiplied = factor.astype(np.float32) * base_image.astype(np.float32)
                # multiplied = (normalize(multiplied) * 255.0).astype(np.uint8)
                multiplied = self.apply_mask(base_image, base_output, "mul", isoline_thresholds=[0.25, 0.5, 0.75, 0.90])

                if "mul" in self.saved_versions:
                    self.write_image(multiplied, correct_prediction, sample_id,
                                     image_name="{}_Mul".format(output_name), modality=modality)

                if "next_to_mul" in self.saved_versions:
                    stacked_images = cv2.hconcat([base_image_3c.astype(multiplied.dtype), multiplied])
                    self.write_image(stacked_images, correct_prediction, sample_id,
                                     image_name="{}_NextToMul".format(output_name), modality=modality)
            # endregion

            # region Additive
            if self._is_saved("additive"):
                # additive = (np.float32(base_image_3c) / 510.0) + (np.float32(output) / 510.0)
                # additive = np.uint8(255 * additive)
                additive = self.apply_mask(base_image, base_output, "add", isoline_thresholds=[0.25, 0.5, 0.75, 0.90])

                if "additive" in self.saved_versions:
                    self.write_image(additive, correct_prediction, sample_id,
                                     image_name="{}_Additive".format(output_name), modality=modality)

                if "next_to_additive" in self.saved_versions:
                    stacked_images = cv2.hconcat([base_image_3c.astype(additive.dtype), additive])
                    self.write_image(stacked_images, correct_prediction, sample_id,
                                     image_name="{}_NextToAdditive".format(output_name), modality=modality)
            # endregion

            # region Segmentation
            if self._is_saved("segmentation"):
                # noinspection PyUnresolvedReferences
                is_background = (output == 0).astype(np.uint8)

                segmented = is_background * base_image_3c + (1.0 - is_background) * output
                if "segmentation" in self.saved_versions:
                    self.write_image(segmented, correct_prediction, sample_id,
                                     image_name="{}_Seg".format(output_name), modality=modality)

                if "next_to_segmentation" in self.saved_versions:
                    stacked_images = cv2.hconcat([base_image_3c.astype(segmented.dtype), segmented])
                    self.write_image(stacked_images, correct_prediction, sample_id,
                                     image_name="{}_NextToSeg".format(output_name), modality=modality)
            # endregion

    def apply_mask(self,
                   image: np.ndarray,
                   mask: np.ndarray,
                   fusion_methods: list[Literal["add", "hue", "mul"]] | Literal["add", "hue", "mul"],
                   isoline_thresholds: list[float] = None
                   ) -> np.ndarray | list[np.ndarray]:
        _fusions = [fusion_methods] if isinstance(fusion_methods, str) else fusion_methods
        image = np.float32(image) / 255.0
        image = ensure_channel_last(image)
        mask = self._format_image(mask, normalize_image=True, rotate=False)
        color_mask = cv2.applyColorMap(mask, self.color_map)
        color_mask = np.float32(color_mask) / 255.0
        mask = np.float32(mask) / mask.max()
        mask = ensure_channel_last(mask)

        if self.mask_threshold is not None:
            below_threshold = np.float32(mask < self.mask_threshold)
        else:
            below_threshold = None

        if isoline_thresholds is not None:
            isolines = np.zeros_like(mask)
            for threshold in isoline_thresholds:
                isoline_base_mask = np.float32(mask > threshold)
                isoline_mask_blurred = cv2.blur(isoline_base_mask, ksize=(4, 4))
                isoline_mask_blurred = ensure_channel_last(isoline_mask_blurred)
                isoline_mask = (isoline_mask_blurred - isoline_base_mask) > 0
                isoline = np.float32(isoline_mask) * threshold
                isolines = np.maximum(isolines, isoline)
            isolines_mask = np.float32(isolines > 0)
            isolines = cv2.applyColorMap(np.uint8(isolines * 255.0), self.color_map)
            isolines = np.float32(isolines) / 255.0
        else:
            isolines_mask, isolines = None, None

        outputs: np.ndarray | list[np.ndarray] = []
        for fusion_method in _fusions:
            if fusion_method == "add":
                output = image * (1.0 - self.overlay_coeff) + color_mask * self.overlay_coeff
            elif fusion_method == "mul":
                output = image * mask
            elif fusion_method == "hue":
                # base_hsv = cv2.cvtColor(base_image_3c.astype(np.float32), cv2.COLOR_BGR2HSV)
                # output_hsv = cv2.cvtColor(output.astype(np.float32), cv2.COLOR_BGR2HSV)
                # base_hsv[..., :2] = output_hsv[..., :2]
                # base_hsv = cv2.cvtColor(base_hsv, cv2.COLOR_HSV2BGR)

                output = cv2.cvtColor(color_mask, cv2.COLOR_BGR2HSV)
                output[..., 2:] = image
                output[..., 1:2] = 0.4
                output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
            else:
                raise RuntimeError("Unknown fusion_method `{}`".format(fusion_method))

            if (below_threshold is not None) and (fusion_method != "mul"):
                output = output * (1.0 - below_threshold) + image * below_threshold

            if isolines is not None:
                output = output * (1.0 - isolines_mask) + isolines * isolines_mask

            output = np.uint8(output * 255.0)
            outputs.append(output)

        if isinstance(fusion_methods, str):
            outputs = outputs[0]

        return outputs

    def write_image(self,
                    image: np.ndarray,
                    correct_prediction: bool | None,
                    sample_id: str | int,
                    image_name: str,
                    modality: Modality | str
                    ) -> None:
        modality = modality.id if isinstance(modality, Modality) else modality
        output_path = self.get_predictions_folder(correct_prediction) / "{}_{}_{}.png".format(sample_id, image_name,
                                                                                              modality)
        cv2.imwrite(output_path.as_posix(), image)

    def _is_saved(self, version: str) -> bool:
        return (version in self.saved_versions) or (("next_to_" + version) in self.saved_versions)

    # endregion

    # region Images (3D)
    def save_images_3d(self,
                       images: MetaTensor | list[MetaTensor],
                       image_ids: list[str | int],
                       correct_predictions: list[bool | None] | None = None
                       ):
        if correct_predictions is None:
            correct_predictions = [None] * len(images)
        for image, image_id, correct_prediction in zip(images, image_ids, correct_predictions):
            self.save_image_3d(image, image_id, correct_prediction)

    def save_image_3d(self,
                      image: MetaTensor,
                      image_id: str | int,
                      correct_prediction: bool | None,
                      ) -> None:
        if self._image_saver is None:
            self._image_saver = SaveImage(output_ext=".mha",
                                          output_dtype=np.float32,
                                          resample=False,
                                          separate_folder=False,
                                          print_log=False)

        filepath = self.get_predictions_folder(correct_prediction=correct_prediction) / str(image_id)
        self._image_saver(image, filename=filepath.as_posix())

    # endregion

    # region Features
    def save_features(self, modality: Modality, features: np.ndarray, ids: list[str], name: str):
        if modality not in self.features_buffer:
            columns = self.features_names.get(modality) or list(range(features.shape[1]))
        else:
            columns = self.features_buffer[modality].columns

        if (len(columns) == 24) and (features.shape[1] == 25):
            columns = columns.insert(18, "PatRisk_degree")

        data_frame = pd.DataFrame(data=features, index=ids, columns=columns)
        if modality in self.features_buffer:
            data_frame = pd.concat([self.features_buffer[modality], data_frame])
        self.features_buffer[modality] = data_frame

        folder = self.visualizer_folder / modality.id
        folder.mkdir(parents=True, exist_ok=True)

        data_frame.index.name = SCAN_ID
        data_frame.to_csv(folder / "scalar_{}.csv".format(name))

        normed = data_frame.values / np.linalg.norm(data_frame, axis=1, keepdims=True)
        normed = pd.DataFrame(normed, index=data_frame.index, columns=columns)
        normed.index.name = SCAN_ID
        normed.to_csv(folder / "normed_scalar_{}.csv".format(name))

        # try:
        radar_figure, _ = radar_from_data_frame(normed, title="Scalar {}".format(name))
        radar_figure.savefig(folder / "scalar_{}.png".format(name))
        # except:
        #     pass

    # endregion

    # region Misc data
    def get_predictions_folder(self, correct_prediction: bool | None):
        if correct_prediction is None:
            folder = self.visualizer_folder
        elif correct_prediction:
            folder = self.correct_predictions_folder
        else:
            folder = self.incorrect_predictions_folder

        if not folder.exists():
            folder.mkdir(parents=True)

        return folder

    def get_ids(self, batch_size: int, ids: list[str] = None) -> list[str]:
        if ids is None:
            ids = ["{:03d}".format(i + self.samples_seen_count) for i in range(batch_size)]
        elif len(ids) != batch_size:
            raise ValueError("The length of inputs and ids must match (got {} and {}).".
                             format(len(ids), batch_size))
        return ids

    # endregion

    # region Fit with train/validation data
    def fit_with_train_batch(self, batch: ModelInput, ids: list[str] = None) -> None:
        raise NotImplementedError

    def fit_with_validation_batch(self, batch: ModelInput, ids: list[str] = None) -> None:
        raise NotImplementedError

    def on_fit_end(self) -> None:
        raise NotImplementedError

    # endregion

    @staticmethod
    def get_correct_predictions(predictions: ClassifierOutput,
                                labels: torch.Tensor | None
                                ) -> list[None] | np.ndarray:
        if (labels is None) or predictions.single_class:
            correct_predictions = [None] * predictions.batch_size
        else:
            predicted_class = torch.argmax(predictions.logits, dim=-1)
            correct_predictions = labels == predicted_class
            correct_predictions = correct_predictions.cpu().numpy()

        return correct_predictions

    # region Properties / Abstract class methods/properties
    # region Outputs folders
    @property
    def visualizer_folder(self) -> Path:
        return self.log_dir / self.visualizer_name

    @property
    def correct_predictions_folder(self) -> Path:
        return self.visualizer_folder / "correct_predictions"

    @property
    def incorrect_predictions_folder(self) -> Path:
        return self.visualizer_folder / "incorrect_predictions"

    # endregion

    # region Fit with train/validation data
    @property
    def should_fit(self) -> bool:
        return self.should_fit_with_train_data or self.should_fit_with_validation_data

    @property
    def should_fit_with_train_data(self) -> bool:
        return False

    @property
    def should_fit_with_validation_data(self) -> bool:
        return False
    # endregion
    # endregion


def ensure_channel_last(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return image

    if len(image.shape) == 2:
        return np.expand_dims(image, axis=-1)

    raise ValueError(image.shape)
