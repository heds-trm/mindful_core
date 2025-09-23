from enum import IntEnum
from typing import Sequence, Iterator, Union, Optional


class ModalityType(IntEnum):
    IMAGE = 2
    SCALAR = 4
    CATEGORICAL = 8
    MASK = 16
    META_DATA = 32
    LABEL = 1

    def to_key(self) -> str:
        return self.name.lower()

    @staticmethod
    def from_key(key: str) -> Optional["ModalityType"]:
        key = key.lower()

        if key.startswith("-"):
            key = key[1:]

        for modality_type in ModalityType:
            if modality_type.to_key() == key:
                return modality_type

        # old name support
        if key == "scan":
            return ModalityType.IMAGE

        return None

    def is_missing(self, data: str | int | Sequence[float] | Sequence[str] | Sequence[int] | None) -> bool:
        if self == ModalityType.LABEL:
            return data is None
        elif self == ModalityType.IMAGE:
            return (data is None) or (len(data) == 0)
        elif self == ModalityType.SCALAR:
            return (data is None) or (len(data) == 0) or all([scalar is None for scalar in data])
        elif self == ModalityType.CATEGORICAL:
            return (data is None) or (len(data) == 0)
        elif self == ModalityType.MASK:
            return (data is None) or (len(data) == 0)
        else:
            raise NotImplementedError(self, type(data))

    @staticmethod
    def convert_and_check_modalities(modalities: list[str]) -> list["ModalityType"]:
        types = [ModalityType.from_key(modality) for modality in modalities]
        undefined = [(modality, modality_type)
                     for modality, modality_type in zip(modalities, types)
                     if modality_type is None]
        if len(undefined) > 0:
            raise ValueError("All modalities must be defined. Undefined modalities: {}.".format(undefined))
        return types


class Modality(object):
    def __init__(self,
                 modality_type: ModalityType | str = None,
                 modality_id: str = None
                 ):
        if (modality_type is None) and (modality_id is None):
            raise ValueError("At least one of `modality_type` and `modality_id` must be provided.")

        if modality_type is None:
            modality_type = modality_id

        if isinstance(modality_type, str):
            key = modality_type
            modality_type = ModalityType.from_key(key)
            if modality_type is None:
                raise ValueError("Incorrect modality type: {}".format(key))

        if modality_id is None:
            modality_id = modality_type.to_key()

        self._type: ModalityType = modality_type
        self._id: str = modality_id

    def __repr__(self) -> str:
        return "{}:{}".format(self._type.to_key(), self.id)

    @staticmethod
    def parse(text: str) -> "Modality":
        if ":" not in text:
            return Modality(modality_type=text)

        sub_strings = text.split(":")
        if len(sub_strings) != 2:
            raise ValueError("Too many subfields in `{}`. Found multiple `:`.".format(text))

        return Modality(*sub_strings)

    @staticmethod
    def is_modality_repr(text: str) -> bool:
        return text.count(":") == 1

    @property
    def id(self) -> str:
        return self._id

    @property
    def type(self) -> ModalityType:
        return self._type

    def __eq__(self, other: "Modality"):
        if not isinstance(other, Modality):
            return False
        return (self._id == other._id) and (self._type == other._type)

    def __hash__(self):
        return hash((self._id, self._type))

    def matches(self, other: Union["Modality", ModalityType, str]) -> bool:
        if isinstance(other, Modality):
            return (other.id == self.id) and (other.type == other.type)

        elif isinstance(other, ModalityType):
            return other == self.type

        elif isinstance(other, str):
            return other == self.id

        else:
            raise TypeError

    @property
    def ignored(self) -> bool:
        return self.id.startswith("-")


class ModalitySet(object):
    def __init__(self, modalities: list[Modality | ModalityType | str]):
        self.modalities: list[Modality] = []
        for modality in modalities:
            self.add(modality)

    @property
    def multimodal(self) -> bool:
        return self.get_modality_count(False, False) > 1

    def get_modality_count(self, include_labels=False, include_masks=False) -> int:
        if include_labels and include_masks:
            return len(self.modalities)

        count = 0
        for modality in self.modalities:
            if (modality.type == ModalityType.LABEL) and include_labels:
                count += 1
            elif (modality.type == ModalityType.MASK) and include_masks:
                count += 1
            else:
                count += 1
        return count

    def __contains__(self, value: Modality | ModalityType | str) -> bool:
        if isinstance(value, Modality) and (value in self.modalities):
            return True

        for modality in self.modalities:
            if modality.matches(value):
                return True

        return False

    def __iter__(self) -> Iterator[Modality]:
        yield from self.modalities

    def __getitem__(self, key: str | int) -> Modality:
        if isinstance(key, int):
            return self.modalities[key]

        for modality in self.modalities:
            if key == modality.id:
                return modality

        raise KeyError(key)

    def __sub__(self, value: Modality | ModalityType | str) -> "ModalitySet":
        if value not in self:
            raise ValueError(value)

        modalities = [modality for modality in self.modalities if not modality.matches(value)]
        return ModalitySet(modalities)

    def add(self, modality: Modality | ModalityType | str) -> None:
        modality = modality if isinstance(modality, Modality) else Modality(modality)
        self.modalities.append(modality)

    def remove(self, modality: Modality | ModalityType | str) -> None:
        modality = modality if isinstance(modality, Modality) else Modality(modality)
        self.modalities.remove(modality)

    def get_index(self, value: Modality | ModalityType | str) -> int:
        for i, modality in enumerate(self.modalities):
            if modality.matches(value):
                return i

        raise ValueError(value)

    def get(self, value: ModalityType | str) -> Modality | None:
        for modality in self.modalities:
            if modality.matches(value):
                return modality
        return None

    @property
    def used_modalities(self) -> "ModalitySet":
        return ModalitySet([modality for modality in self.modalities if not modality.ignored])

    @property
    def ignored_modalities(self) -> "ModalitySet":
        return ModalitySet([modality for modality in self.modalities if modality.ignored])

    @property
    def ids(self) -> list[str]:
        return [modality.id for modality in self.modalities]

    def get_modality_type_indices(self, modality_type: ModalityType) -> list[int]:
        indices = [i for i, modality in enumerate(self) if modality.type == modality_type]
        return indices
