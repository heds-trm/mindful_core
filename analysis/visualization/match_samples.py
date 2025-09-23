import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from enum import Enum

from data import ModalitySet
from models.model_output import ModelOutput, ClassifierOutput, PrototypeOutput, RepresentationOutput
from analysis.visualization.visualizer import Visualizer
from models.module import MindfulModule, ModelInput, unbatch_model_inputs
from models.classification.prototype_layer import PrototypeLayer


class MatchMode(Enum):
    REPRESENTATION = 1
    PROTOTYPE = 2

    def to_label(self) -> str:
        if self == MatchMode.REPRESENTATION:
            return "representation"
        if self == MatchMode.PROTOTYPE:
            return "prototype"
        raise NotImplementedError("Match mode `{}` is not implemented".format(self))


class MatchReference(object):
    def __init__(self, sample_id: str, sample_data: ModelInput) -> None:
        self.sample_id = sample_id
        self.sample_data = sample_data

    @staticmethod
    def from_batch(batch: ModelInput, ids: list[str]) -> list["MatchReference"]:
        return [MatchReference(sample_id, sample_data)
                for sample_id, sample_data
                in zip(ids, unbatch_model_inputs(batch))]


class MatchSamples(Visualizer):
    visualizer_name = "match_samples"

    def __init__(self,
                 model: MindfulModule,
                 log_dir: str | Path,
                 input_modalities: ModalitySet = None,
                 matched_subset="train",
                 k=1,  # number of closest samples taken
                 **kwargs
                 ):
        super().__init__(model=model,
                         log_dir=log_dir,
                         saved_slices=None,
                         saved_versions=None,
                         color_map=None,
                         mask_threshold=None,
                         overlay_coeff=None,
                         resize_method=None,
                         input_modalities=input_modalities
                         )
        self.matched_subset = matched_subset
        self.match_mode = self.infer_match_mode(model)

        if self.match_mode == MatchMode.REPRESENTATION:
            reference_representations = []
        elif self.match_mode == MatchMode.PROTOTYPE:
            reference_representations = None
        else:
            raise NotImplementedError("Match mode `{}` is not implemented".format(self.match_mode))

        self.reference_representations: torch.Tensor | list[torch.Tensor] | None = reference_representations
        self.reference_data: list[MatchReference] = []

        if k != 1:
            raise NotImplementedError

        self.matches: dict[str, str] = {}
        self._kwargs = kwargs

    def __call__(self,
                 sample: list[torch.Tensor],
                 **kwargs
                 ) -> dict[str, torch.Tensor]:
        batched_sample = [modality.unsqueeze(0) for modality in sample]
        match = self._run_batched(batched_sample)[0]
        match_data = [match.sample_data] if isinstance(match.sample_data, torch.Tensor) else match.sample_data
        return {modality.id: modality_value
                for modality, modality_value
                in zip(self.input_modalities, match_data)}

    def run_batch(self, batch: ModelInput, ids: list[str] = None):
        matches: list[MatchReference] = self._run_batched(batch)
        for sample_id, match_reference in zip(ids, matches):
            self.matches[sample_id] = match_reference.sample_id

    def finalize(self) -> None:
        matches = [{"Sample ID": sample_id, "Match ID": match_id} for sample_id, match_id in self.matches.items()]
        data_frame = pd.DataFrame(matches)
        data_frame.to_csv(self.get_predictions_folder(None) / "matches.csv", index=False)

    def _run_batched(self, batch: ModelInput) -> list[MatchReference]:
        match_vector = self.get_match_vector(batch)

        if self.match_mode == MatchMode.PROTOTYPE:
            closest_indices = self.run_representation_matching(match_vector)
        elif self.match_mode == MatchMode.REPRESENTATION:
            closest_indices = self.run_prototype_matching(match_vector)
        else:
            raise NotImplementedError("Match mode `{}` is not implemented".format(self.match_mode))

        matches = [self.reference_data[i] for i in closest_indices]
        return matches

    def get_match_vector(self, inputs: ModelInput) -> torch.Tensor:
        with torch.no_grad():
            model_outputs: ModelOutput = self.model(inputs)

        return self.extract_match_vector(model_outputs)

    def extract_match_vector(self, model_outputs: ModelOutput) -> torch.Tensor:
        if self.match_mode == MatchMode.PROTOTYPE:
            if isinstance(model_outputs, ClassifierOutput):
                if not model_outputs.has_prototype_outputs:
                    raise RuntimeError("Missing prototype outputs in classifier outputs.")

                prototype_outputs = model_outputs.prototype_outputs
            elif isinstance(model_outputs, PrototypeOutput):
                prototype_outputs = model_outputs
            else:
                raise TypeError(type(model_outputs))

            return prototype_outputs.similarities

        if self.match_mode == MatchMode.REPRESENTATION:
            if isinstance(model_outputs, RepresentationOutput):
                return model_outputs.representations
            elif isinstance(model_outputs, ClassifierOutput):
                if not model_outputs.has_intermediate_outputs:
                    raise RuntimeError("Missing representations in classifier outputs.")
                return model_outputs.intermediate_outputs
            else:
                raise TypeError(type(model_outputs))

        raise NotImplementedError("Match mode `{}` is not implemented".format(self.match_mode))

    def run_representation_matching(self, batch: torch.Tensor) -> list[int]:
        batch = batch.unsqueeze(dim=1)
        # noinspection PyUnresolvedReferences
        reference_representations = self.reference_representations.unsqueeze(dim=0)
        distance = torch.square(batch - reference_representations).sum(dim=-1)
        batch_closest_indices = distance.argmin(dim=1).tolist()
        return batch_closest_indices

    @staticmethod
    def run_prototype_matching(batch: torch.Tensor) -> list[int]:
        return batch.argmax(dim=1).tolist()

    # region Fit
    def fit_on_batch(self, batch: ModelInput, ids: list[str] = None) -> None:
        match_vector = self.get_match_vector(batch)
        batch_reference_data = MatchReference.from_batch(batch, ids)

        if self.match_mode == MatchMode.REPRESENTATION:
            self.reference_representations.append(match_vector)
            self.reference_data += batch_reference_data

        elif self.match_mode == MatchMode.PROTOTYPE:
            batch_similarities, batch_closest_indices = match_vector.max(dim=0)
            batch_closest_indices = batch_closest_indices.tolist()

            if self.reference_representations is None:
                self.reference_representations = batch_similarities
                self.reference_data = [batch_reference_data[i] for i in batch_closest_indices]
            else:
                is_new_closest = batch_similarities > self.reference_representations
                # noinspection PyTypeChecker
                self.reference_representations = torch.where(is_new_closest, batch_similarities,
                                                             self.reference_representations)
                # noinspection PyTypeChecker, PyUnresolvedReferences
                new_best_indices = torch.arange(len(is_new_closest), device=is_new_closest.device)[is_new_closest]
                for i in new_best_indices.tolist():
                    self.reference_data[i] = batch_reference_data[batch_closest_indices[i]]

    def fit_with_train_batch(self, batch: ModelInput, ids: list[str] = None) -> None:
        return self.fit_on_batch(batch, ids)

    def fit_with_validation_batch(self, batch: ModelInput, ids: list[str] = None) -> None:
        return self.fit_on_batch(batch, ids)

    def on_fit_end(self) -> None:
        if self.match_mode == MatchMode.REPRESENTATION:
            self.reference_representations = torch.concat(self.reference_representations, dim=0)

    @property
    def should_fit_with_train_data(self) -> bool:
        return self.matched_subset == "train"

    @property
    def should_fit_with_validation_data(self) -> bool:
        return not self.should_fit_with_train_data

    # endregion

    @staticmethod
    def infer_match_mode(model: nn.Module) -> MatchMode:
        if PrototypeLayer.model_has_prototype_layer(model):
            return MatchMode.PROTOTYPE

        return MatchMode.REPRESENTATION
