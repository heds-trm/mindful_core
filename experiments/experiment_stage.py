from abc import ABCMeta, abstractmethod
import inspect
from typing import Type, Any, Iterable

from mindful_core.utils.struct import AliasDict


class ExperimentStage(metaclass=ABCMeta):
    _stage_register: AliasDict[str, Type["ExperimentStage"]] = AliasDict()

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not inspect.isabstract(cls):
            identifier = cls.stage_identifier()
            cls._stage_register[identifier] = cls

            aliases = cls.stage_aliases()
            if len(aliases) > 0:
                cls._stage_register.add_aliases(identifier, *aliases)

    @classmethod
    @abstractmethod
    def stage_identifier(cls) -> str:
        raise NotImplementedError("Method `stage_identifier` must be implemented in subclasses.")

    @classmethod
    def stage_aliases(cls) -> list[str]:
        return []

    @classmethod
    def all_stage_identifiers(cls) -> Iterable[str]:
        return cls._stage_register.keys_and_aliases()

    @classmethod
    def get_stage_class(cls, stage_identifier: str) -> Type["ExperimentStage"]:
        return cls._stage_register[stage_identifier]

    # noinspection PyMethodMayBeStatic
    def serialize(self) -> dict:
        return {}


class TrainStage(ExperimentStage):
    @classmethod
    def stage_identifier(cls) -> str:
        return "train"


class TestStage(ExperimentStage):
    @classmethod
    def stage_identifier(cls) -> str:
        return "test"


class VisualisationStage(ExperimentStage):
    @classmethod
    def stage_identifier(cls) -> str:
        return "visualisation"

    @classmethod
    def stage_aliases(cls) -> list[str]:
        return ["visualization", "visualise", "visualize"]


class PseudoLabellingStage(ExperimentStage):
    @classmethod
    def stage_identifier(cls) -> str:
        return "pseudo_labelling"

    @classmethod
    def stage_aliases(cls) -> list[str]:
        return ["pseudo", "pseudolabelling", "pseudo labelling", "pseudo-labelling"]


class ConfidenceAnalysisStage(ExperimentStage):
    @classmethod
    def stage_identifier(cls) -> str:
        return "confidence_analysis"

    @classmethod
    def stage_aliases(cls) -> list[str]:
        return ["confidence", "confidence analysis", "confidence-analysis"]


class RepresentationStage(ExperimentStage):
    @classmethod
    def stage_identifier(cls) -> str:
        return "representation"


class SegmentationStage(ExperimentStage):
    @classmethod
    def stage_identifier(cls) -> str:
        return "segmentation"


class ExperimentStages(object):
    def __init__(self, stages_configs: str | list[str] | dict[str, Any]) -> None:
        if isinstance(stages_configs, str):
            reference = stages_configs
            found_stages = {}
            for stage_identifier in ExperimentStage.all_stage_identifiers():
                if stage_identifier not in stages_configs:
                    continue

                stage_class = ExperimentStage.get_stage_class(stage_identifier)

                if stage_class in found_stages:
                    continue

                found_stages[stage_class] = stage_identifier

            stages_configs = list(found_stages.values())
            stages_configs.sort(key=lambda stage_id: reference.index(stage_id))

        stages: list[ExperimentStage] = []
        for stage_identifier in stages_configs:
            stage_config = stages_configs[stage_identifier] if isinstance(stages_configs, dict) else {}
            stage_class = ExperimentStage.get_stage_class(stage_identifier)
            stage = stage_class(**stage_config)
            stages.append(stage)

        self.stages = stages

    def _include_stage(self, stage_type: Type[ExperimentStage]):
        return any([isinstance(stage, stage_type) for stage in self.stages])

    @property
    def include_training(self) -> bool:
        return self._include_stage(TrainStage)

    @property
    def includes_testing(self) -> bool:
        return self._include_stage(TestStage)

    @property
    def includes_pseudo_labelling(self) -> bool:
        return self._include_stage(PseudoLabellingStage)

    @property
    def includes_confidence_analysis(self) -> bool:
        return self._include_stage(ConfidenceAnalysisStage)

    @property
    def includes_visualization(self) -> bool:
        return self._include_stage(VisualisationStage)

    @property
    def includes_representation(self) -> bool:
        return self._include_stage(RepresentationStage)

    @property
    def includes_segmentation(self) -> bool:
        return self._include_stage(SegmentationStage)

    def serialize(self) -> dict:
        return {stage.stage_identifier(): stage.serialize() for stage in self.stages}
