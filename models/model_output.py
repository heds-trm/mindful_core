import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from abc import abstractmethod, ABC
from typing import Iterator, Self

from mindful_core.utils.tensor_utils import lerp
from mindful_core.data.modalities import ModalitySet, Modality, ModalityType


class ModelOutput(ABC):
    """
    Abstract data class for grouping and for streamlining various outputs of Mindful models.

    :param batched: Required to explicitly differentiate between batched and unbatched data. Defaults to True.
    """

    def __init__(self, batched=True) -> None:
        self.batched = batched

    @abstractmethod
    def distance(self, other: Self) -> torch.Tensor:
        """
        Measures the distance between two model outputs.
        Use cases include measuring the sensitivity of a model's outputs to its inputs.

        :param other: Other ModelOutput to measure distance against.
        :returns: A tensor containing the measure of the distance between the two model outputs.
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[Self]:
        """
        Iterate over the samples contained in this batch of ModelOutputs.

        :returns: An unbatched ModelOutput.
        :raises RuntimeError: Raised if this ModelOutput is not a batch.
        """
        if not self.batched:
            raise RuntimeError("Cannot iterate on a single instance.")

        for i in range(self.batch_size):
            yield self[i]

    @abstractmethod
    def __getitem__(self, index) -> Self:
        """
        Returns the sample at the given index of the batch. Un-batches the ModelOutput in the process.

        :returns: An unbatched ModelOutput.
        :raises RuntimeError: Raised if this ModelOutput is not a batch.
        :raises IndexError: Raised if index is out of bounds.
        """
        raise NotImplementedError

    @abstractmethod
    def as_batch(self) -> Self:
        """
        Turns an unbatched ModelOutput into a batched ModelOutput with a single element in it.

        :returns: A batched ModelOutput.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def list_to_batch(cls, outputs: list[Self]) -> Self:
        """
        Concatenates the different ModelOutput into a single batch. Elements of the list may or may not be batched and
        will be harmonized.

        :returns: A batched ModelOutput.
        :raises ValueError: If concatenation could not be performed (e.g., empty list).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """
        :returns: The size of the batch contained in this ModelOutput. Returns -1 if unbatched.
        """
        raise NotImplementedError

    @staticmethod
    def sync_batched(first: Self, second: Self) -> tuple[Self, Self]:
        """
        Harmonizes the batch status of both ModelOutputs. Required for operations such as comparisons.

        :param first: First ModelOutput for which to sync batch status.
        :param second: Second ModelOutput for which to sync batch status.
        :returns: The two ModelOutput with their batch status now matching.
        """
        if first.batched or second.batched:
            if not second.batched:
                second = second.as_batch()
            elif not first.batched:
                first = first.as_batch()
        return first, second


class PrototypeOutput(ModelOutput):
    """
    /!\\\\ W.I.P. data class /!\\\\.

    Intended for matching samples to find the N closest prototypes in a reference set, allowing for
    tasks such as sample retrieval or explainability.

    :param prototype_distances: Measure of the distance between samples and prototypes. 1D or 2D (batched) tensor.
    :param similarities: Measure of the similarity between samples and prototypes. 1D or 2D (batched) tensor.
    May not be strictly proportional to `prototype_distances`.
    """

    def __init__(self,
                 prototype_distances: torch.Tensor,
                 similarities: torch.Tensor
                 ) -> None:
        batched = len(prototype_distances.shape) > 1
        super().__init__(batched)

        self.prototype_distances = prototype_distances
        self.similarities = similarities

    def distance(self, other: Self) -> torch.Tensor:
        first, other = self.sync_batched(self, other)
        first: PrototypeOutput
        other: PrototypeOutput

        return torch.abs(first.similarities - other.similarities)

    def __getitem__(self, index) -> Self:
        if not self.batched:
            raise RuntimeError("This output only contains a single instance.")

        if not (0 <= index < self.prototype_distances.shape[0]):
            raise IndexError("Invalid index {} (expected to be between 0 and {})".
                             format(index, self.prototype_distances.shape[0] - 1))

        prototype_distances = self.prototype_distances[index]
        similarities = self.similarities[index]
        return PrototypeOutput(prototype_distances, similarities)

    def as_batch(self) -> Self:
        if self.batched:
            return self

        prototype_distances = self.prototype_distances.unsqueeze(0)
        similarities = self.similarities.unsqueeze(0)
        return PrototypeOutput(prototype_distances, similarities)

    @classmethod
    def list_to_batch(cls, outputs: list[Self]) -> Self:
        # region check consistency
        if len(outputs) == 0:
            raise ValueError("The list must contain at least one element to allow batching.")

        # endregion

        outputs = [output.as_batch() for output in outputs]
        prototype_distances = torch.stack([output.prototype_distances for output in outputs], dim=0)
        similarities = torch.stack([output.similarities for output in outputs], dim=0)
        return PrototypeOutput(prototype_distances, similarities)

    @property
    def batch_size(self) -> int:
        return len(self.prototype_distances) if self.batched else -1

    @staticmethod
    def concat(outputs: list[Self]) -> Self:
        if len(outputs) == 0:
            raise ValueError("The sequence must contain at least one element.")

        prototype_distances = torch.concat([output.prototype_distances for output in outputs], dim=0)
        similarities = torch.concat([output.similarities for output in outputs], dim=0)

        return PrototypeOutput(prototype_distances, similarities)


# TODO: Allow the type of `intermediate_outputs` to be RepresentationOutput (move this class definition first)
#   Note: Not all intermediate_outputs are representations (e.g. ensemble model invidual outputs)

class ClassifierOutput(ModelOutput):
    """
    Generic data class handling the outputs of classification models.

    :param single_class: Whether logits are single-class logits (usually followed by a sigmoid)
        or multi-class (usually followed by a softmax or an argmax).
    :param logits: Class logits yielded by the model.
    :param confidence: [Optional] A confidence score (if batched, for each sample in the batch).
    :param confidence_threshold: [Optional] Associated `confidence` threshold for discriminating out-of-distribution
        samples.
    :param intermediate_outputs: [Optional] Intermediate outputs yielded by the model during inference
        (e.g., representations).
    :param prototype_outputs: [Optional] PrototypeOutput associated with the samples.
    :param batched: Required to explicitly differentiate between batched and unbatched data. Defaults to True.
    """

    def __init__(self,
                 single_class: bool,
                 logits: torch.Tensor,
                 confidence: torch.Tensor | None = None,
                 confidence_threshold: torch.Tensor | float | None = None,
                 intermediate_outputs: torch.Tensor | None = None,
                 prototype_outputs: PrototypeOutput | None = None,
                 batched: bool = True) -> None:
        super().__init__(batched)
        self.single_class = single_class
        squeeze_logits = single_class and (len(logits.shape) > 1)
        self.logits = torch.squeeze(logits, dim=-1) if squeeze_logits else logits
        self.confidence = confidence
        self.intermediate_outputs = intermediate_outputs
        self.prototype_outputs = prototype_outputs

        if ((confidence_threshold is not None)
                and isinstance(confidence_threshold, torch.Tensor)
                and (confidence_threshold.numel() == 1)):
            confidence_threshold = float(confidence_threshold.detach())
        self.confidence_threshold: torch.Tensor | float | None = confidence_threshold

    def distance(self, other: Self) -> torch.Tensor:
        first, other = self.sync_batched(self, other)
        first: ClassifierOutput
        other: ClassifierOutput

        result = other.get_probabilities() - first.get_probabilities()
        if self.single_class:
            result = torch.abs(result)
        else:
            result = torch.sqrt(torch.sum(torch.square(result), dim=-1))

        return result

    def __getitem__(self, index) -> Self:
        if not self.batched:
            raise RuntimeError("This output only contains a single instance.")

        if not (0 <= index < self.logits.shape[0]):
            raise IndexError("Invalid index {} (expected to be between 0 and {})".
                             format(index, self.logits.shape[0] - 1))

        if self.has_confidence:
            confidence = self.confidence[index]
            if self.confidence_threshold is None:
                confidence_threshold = None
            elif self.has_unique_confidence_threshold:
                confidence_threshold = self.confidence_threshold
            elif isinstance(self.confidence_threshold, torch.Tensor):
                confidence_threshold = self.confidence_threshold[index]
            else:
                raise RuntimeError("Unexpected type for `self.confidence_threshold`: {}".
                                   format(type(self.confidence_threshold)))
        else:
            confidence = None
            confidence_threshold = None

        intermediate_outputs = None if self.intermediate_outputs is None else self.intermediate_outputs[index]
        prototype_outputs = None if self.prototype_outputs is None else self.prototype_outputs[index]
        return ClassifierOutput(single_class=self.single_class,
                                logits=self.logits[index],
                                confidence=confidence,
                                confidence_threshold=confidence_threshold,
                                intermediate_outputs=intermediate_outputs,
                                prototype_outputs=prototype_outputs,
                                batched=False)

    def as_batch(self) -> Self:
        if self.batched:
            return self
        else:
            confidence = None if self.confidence is None else self.confidence.unsqueeze(0)
            intermediate_outputs = (None if self.intermediate_outputs is None
                                    else torch.unsqueeze(self.intermediate_outputs, 0))
            prototype_outputs = None if self.prototype_outputs is None else self.prototype_outputs.as_batch()
            return ClassifierOutput(single_class=self.single_class,
                                    logits=self.logits.unsqueeze(0),
                                    confidence=confidence,
                                    confidence_threshold=self.confidence_threshold,
                                    intermediate_outputs=intermediate_outputs,
                                    prototype_outputs=prototype_outputs,
                                    batched=True)

    @classmethod
    def list_to_batch(cls, outputs: list[Self]) -> Self:
        # region check consistency
        if len(outputs) == 0:
            raise ValueError("The list must contain at least one element to allow batching.")

        single_class = outputs[0].single_class
        if not all([output.single_class == single_class for output in outputs]):
            raise ValueError("All outputs should either be single-class or multi-class, not a mix of both.")

        has_confidence = outputs[0].confidence is not None
        if not all([output.has_confidence == has_confidence for output in outputs]):
            raise ValueError("Either all outputs or none should have confidence values, not a mix of both.")

        has_intermediate = outputs[0].has_intermediate_outputs
        if not all([output.has_intermediate_outputs == has_intermediate for output in outputs]):
            raise ValueError("Either all outputs or none should have intermediate outputs, not a mix of both.")

        has_prototype_outputs = outputs[0].has_prototype_outputs
        if not all([output.has_prototype_outputs == has_prototype_outputs for output in outputs]):
            raise ValueError("Either all outputs or none should have prototype outputs, not a mix of both.")
        # endregion

        outputs = [output.as_batch() for output in outputs]
        logits = torch.concat([output.logits for output in outputs])
        confidence, confidence_threshold = cls.concat_confidence_values(outputs)

        if has_intermediate:
            intermediate_outputs = torch.concat([output.intermediate_outputs for output in outputs])
        else:
            intermediate_outputs = None

        if has_prototype_outputs:
            prototype_outputs = PrototypeOutput.list_to_batch([output.prototype_outputs for output in outputs])
        else:
            prototype_outputs = None

        return ClassifierOutput(single_class=single_class,
                                logits=logits,
                                confidence=confidence,
                                confidence_threshold=confidence_threshold,
                                intermediate_outputs=intermediate_outputs,
                                prototype_outputs=prototype_outputs,
                                batched=True)

    @property
    def batch_size(self) -> int:
        return len(self.logits) if self.batched else -1

    @property
    def has_confidence(self) -> bool:
        return self.confidence is not None

    @property
    def has_confidence_threshold(self) -> bool:
        return self.confidence_threshold is not None

    @property
    def has_unique_confidence_threshold(self) -> bool:
        if not self.has_confidence_threshold:
            return False

        if isinstance(self.confidence_threshold, float):
            return True

        if isinstance(self.confidence_threshold, torch.Tensor):
            return self.confidence_threshold.numel() == 1
        else:
            raise RuntimeError("Unexpected type for `self.confidence_threshold`: {}".
                               format(type(self.confidence_threshold)))

    @property
    def has_intermediate_outputs(self) -> bool:
        return self.intermediate_outputs is not None

    @property
    def has_prototype_outputs(self) -> bool:
        return self.prototype_outputs is not None

    def get_probabilities(self) -> torch.Tensor:
        if self.single_class:
            probabilities = torch.sigmoid(self.logits)
        else:
            probabilities = torch.softmax(self.logits, dim=-1)
        return probabilities

    def get_predictions(self, labels: torch.Tensor) -> torch.Tensor:
        if self.confidence is None:
            return self.logits
        else:
            return self.get_confidence_predictions(labels)

    def get_confidence_predictions(self, labels: torch.Tensor) -> torch.Tensor:
        probabilities = self.get_probabilities()

        random_mask = torch.bernoulli(torch.full(self.confidence.size(), 0.5, device=self.confidence.device))
        masked_confidence = lerp(torch.sigmoid(self.confidence), 1.0, random_mask)
        if self.single_class:
            probabilities = lerp(labels, probabilities, masked_confidence)
        else:
            class_count = self.logits.size(-1)
            one_hot_labels = nn.functional.one_hot(labels.to(torch.int64), class_count).to(self.logits.dtype)
            masked_confidence = masked_confidence.unsqueeze(-1).expand_as(one_hot_labels)
            probabilities = lerp(one_hot_labels, probabilities, masked_confidence)

        return probabilities

    def detached_outputs(self) -> dict[str, torch.Tensor]:
        outputs = {"predictions": self.logits.detach()}
        if self.confidence is not None:
            outputs["confidence"] = self.confidence.detach()
        return outputs

    @staticmethod
    def concat_logits(outputs: list[Self]) -> torch.Tensor:
        return torch.concat([batch_outputs.logits for batch_outputs in outputs], dim=0)

    @staticmethod
    def concat(outputs: list[Self]) -> Self:
        if len(outputs) == 0:
            raise ValueError("The sequence must contain at least one element.")
        single_class = outputs[0].single_class
        logits = torch.concat([batch_outputs.logits for batch_outputs in outputs], dim=0)

        confidence, confidence_threshold = ClassifierOutput.concat_confidence_values(outputs)

        if outputs[0].has_intermediate_outputs:
            intermediate_outputs = torch.concat([batch_outputs.intermediate_outputs
                                                 for batch_outputs in outputs], dim=0)
        else:
            intermediate_outputs = None

        if outputs[0].has_prototype_outputs:
            prototype_outputs = PrototypeOutput.concat([batch_outputs.prototype_outputs for batch_outputs in outputs])
        else:
            prototype_outputs = None

        return ClassifierOutput(single_class=single_class,
                                logits=logits,
                                confidence=confidence,
                                confidence_threshold=confidence_threshold,
                                intermediate_outputs=intermediate_outputs,
                                prototype_outputs=prototype_outputs)

    @staticmethod
    def concat_confidence_values(outputs: list[Self]
                                 ) -> tuple[torch.Tensor | None, torch.Tensor | float | None]:
        if not all([output.has_confidence for output in outputs]):
            return None, None

        confidence = torch.concat([batch_outputs.confidence for batch_outputs in outputs], dim=0)
        confidence_threshold = ClassifierOutput.concat_confidence_thresholds(outputs)
        return confidence, confidence_threshold

    @staticmethod
    def concat_confidence_thresholds(outputs: list[Self]) -> torch.Tensor | float | None:
        if not all([output.has_confidence_threshold for output in outputs]):
            return None

        concat_confidence_threshold = False
        if not all([output.has_unique_confidence_threshold] for output in outputs):
            concat_confidence_threshold = True

        confidence_threshold = float(outputs[0].confidence_threshold)
        if not all([output.confidence_threshold == confidence_threshold] for output in outputs):
            concat_confidence_threshold = True

        if concat_confidence_threshold:
            thresholds = [torch.as_tensor(output.confidence_threshold).view(-1) for output in outputs]
            confidence_threshold = torch.concat(thresholds)

        return confidence_threshold


class RepresentationOutput(ModelOutput):
    """
    Generic data class for handling representations / intermediate outputs.

    :param representations: Representation tensors.
    :param intermediate_outputs: [Optional] Intermediate outputs the model may have produced during inference.
    :param batched: Required to explicitly differentiate between batched and unbatched data. Defaults to True.
    """

    def __init__(self,
                 representations: torch.Tensor,
                 intermediate_outputs=None,
                 batched=True) -> None:
        super().__init__(batched)
        self.representations = representations
        self.intermediate_outputs = intermediate_outputs

    def distance(self, other: Self) -> torch.Tensor:
        first, other = self.sync_batched(self, other)
        first: RepresentationOutput
        other: RepresentationOutput

        start_dim = 1 if self.batched else 0
        dims = list(range(start_dim, len(self.representations.shape)))

        return torch.linalg.norm(other.representations - first.representations, dim=dims)

    def __getitem__(self, index) -> Self:
        if not self.batched:
            raise RuntimeError("This output only contains a single instance.")

        if not (0 <= index < self.representations.shape[0]):
            raise IndexError("Invalid index {} (expected to be between 0 and {})".
                             format(index, self.representations.shape[0] - 1))

        intermediate_outputs = None if self.intermediate_outputs is None else self.intermediate_outputs[index]
        return RepresentationOutput(representations=self.representations[index],
                                    intermediate_outputs=intermediate_outputs,
                                    batched=False)

    def as_batch(self) -> Self:
        if self.batched:
            return self
        else:
            intermediate_outputs = (None if self.intermediate_outputs is None
                                    else torch.unsqueeze(self.intermediate_outputs, 0))
            return RepresentationOutput(representations=self.representations.unsqueeze(0),
                                        intermediate_outputs=intermediate_outputs,
                                        batched=True)

    @classmethod
    def list_to_batch(cls, outputs: list[Self]) -> Self:
        # region check consistency
        if len(outputs) == 0:
            raise ValueError("The list must contain at least one element to allow batching.")

        has_intermediate = outputs[0].intermediate_outputs is not None
        if not all([output.has_intermediate_outputs == has_intermediate for output in outputs]):
            raise ValueError("Either all outputs or none should have intermediate outputs, not a mix of both.")
        # endregion

        outputs = [output.as_batch() for output in outputs]
        representations = torch.concat([output.representations for output in outputs])
        intermediate_outputs = (torch.concat([output.intermediate_outputs for output in outputs])
                                if has_intermediate else None)

        return RepresentationOutput(representations=representations,
                                    intermediate_outputs=intermediate_outputs,
                                    batched=True)

    @property
    def batch_size(self) -> int:
        return len(self.representations) if self.batched else -1

    @property
    def has_intermediate_outputs(self) -> bool:
        return self.intermediate_outputs is not None

    def detached_representations(self) -> torch.Tensor:
        return self.representations.detach()


class BoundingBoxOutput(ModelOutput):
    """
    /!\\\\ W.I.P. data class /!\\\\.

    Data class for segmentation models that output bounding boxes (with class logits).

    :param regression: The bounding box regressor outputs, representing the position and size of the bounding box(es).
    :param logits: Class logits associated with the bounding box(es).
    :param batched: Required to explicitly differentiate between batched and unbatched data. Defaults to True.
    """

    def __init__(self,
                 regression: torch.Tensor,
                 logits: torch.Tensor,
                 batched=True) -> None:
        super().__init__(batched)

        self.regression = regression
        self.logits = logits

    def distance(self, other) -> torch.Tensor:
        first, other = self.sync_batched(self, other)
        first: ClassifierOutput
        other: ClassifierOutput

        result = other.get_probabilities() - first.get_probabilities()
        result = torch.sqrt(torch.sum(torch.square(result), dim=-1))

        return result

    def __getitem__(self, index) -> Self:
        if not self.batched:
            raise RuntimeError("This output only contains a single instance.")

        if not (0 <= index < self.regression.shape[0]):
            raise IndexError("Invalid index {} (expected to be between 0 and {})".
                             format(index, self.regression.shape[0] - 1))

        return BoundingBoxOutput(self.regression[index], self.logits[index], batched=False)

    def as_batch(self) -> Self:
        if self.batched:
            return self
        else:
            return BoundingBoxOutput(regression=self.regression.unsqueeze(0),
                                     logits=self.logits.unsqueeze(0),
                                     batched=True)

    @classmethod
    def list_to_batch(cls, outputs: list[Self]) -> Self:
        # region check consistency
        if len(outputs) == 0:
            raise ValueError("The list must contain at least one element to allow batching.")

        # endregion

        outputs = [output.as_batch() for output in outputs]
        regression = torch.concat([output.regression for output in outputs])
        logits = torch.concat([output.logits for output in outputs])

        return BoundingBoxOutput(regression=regression,
                                 logits=logits,
                                 batched=True)

    @property
    def batch_size(self) -> int:
        return len(self.regression) if self.batched else -1

    def detached_outputs(self) -> dict[str, torch.Tensor]:
        return {
            "regression": self.regression.detach(),
            "logits": self.logits.detach(),
        }


class SegmentationOutput(ModelOutput):
    """
    Data class for segmentation models that output segmentation maps.

    :param single_class: Whether logits are single-class logits (usually followed by a sigmoid)
        or multi-class (usually followed by a softmax or an argmax).
    :param logits: Element-wise class logits (e.g., pixelwise class logits for images).
    :param batched: Required to explicitly differentiate between batched and unbatched data. Defaults to True.
    """

    def __init__(self,
                 single_class: bool,
                 logits: torch.Tensor,
                 batched: bool = True
                 ) -> None:
        super().__init__(batched)
        self.single_class = single_class
        self.logits = torch.squeeze(logits, dim=-1) if single_class else logits

    def get_probabilities(self) -> torch.Tensor:
        if self.single_class:
            probabilities = torch.sigmoid(self.logits)
        else:
            probabilities = torch.softmax(self.logits, dim=-1)
        return probabilities

    def distance(self, other) -> torch.Tensor:
        raise NotImplementedError

    def __getitem__(self, index) -> Self:
        if not self.batched:
            raise RuntimeError("This output only contains a single instance.")

        if not (0 <= index < self.logits.shape[0]):
            raise IndexError("Invalid index {} (expected to be between 0 and {})".
                             format(index, self.logits.shape[0] - 1))

        return SegmentationOutput(single_class=self.single_class,
                                  logits=self.logits[index],
                                  batched=False)

    def as_batch(self) -> Self:
        raise NotImplementedError

    @classmethod
    def list_to_batch(cls, outputs: list[Self]) -> Self:
        raise NotImplementedError

    @staticmethod
    def concat_segmentations(outputs: list[Self]) -> torch.Tensor:
        return torch.concat([batch_outputs.logits for batch_outputs in outputs], dim=0)

    @property
    def batch_size(self) -> int:
        return len(self.logits) if self.batched else -1


class GeneratorOutput(ModelOutput):
    """
    /!\\\\ W.I.P. data class /!\\\\.

    :param data:
    :param modalities:
    :param batched: Required to explicitly differentiate between batched and unbatched data. Defaults to True.
    """

    def __init__(self,
                 data: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
                 modalities: ModalitySet | Modality | ModalityType | list[Modality] | ModalityType,
                 batched=True) -> None:
        super().__init__(batched)

        self.data = data
        if not isinstance(modalities, ModalitySet):
            if isinstance(modalities, (Modality, ModalitySet, str)):
                modalities = [modalities]
            modalities = ModalitySet(modalities)

        self.modalities: ModalitySet = modalities

    def distance(self, other: Self) -> torch.Tensor:
        if self.modalities != other.modalities:
            raise ValueError("Modalities between GeneratorOutputs do not match, cannot measure distance.")

    def __getitem__(self, index) -> Self:
        pass

    def as_batch(self) -> Self:
        pass

    @classmethod
    def list_to_batch(cls, outputs: list[Self]) -> Self:
        pass

    @property
    def batch_size(self) -> int:
        if not self.batched:
            return -1

        data = self.data[0] if self.is_multimodal else self.data
        return data.shape[0]

    @property
    def is_multimodal(self) -> bool:
        return len(self.modalities) > 1
