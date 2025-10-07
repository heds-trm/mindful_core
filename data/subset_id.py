from enum import Enum

from mindful_core.utils.data_constants import SUBSET_ID


class SubsetID(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 4

    def as_prefix(self) -> str:
        if self == SubsetID.TRAIN:
            return "train"
        elif self == SubsetID.VALIDATION:
            return "validation"
        elif self == SubsetID.TEST:
            return "test"
        else:
            raise RuntimeError

    @staticmethod
    def parse(expression: str) -> SUBSET_ID:
        if not isinstance(expression, str):
            raise ValueError("Expression must be a string, got {}.".format(type(expression)))

        expression = expression.lower()
        if expression == "train":
            return SubsetID.TRAIN
        elif expression == "validation":
            return SubsetID.VALIDATION
        elif expression == "test":
            return SubsetID.TEST
        else:
            raise ValueError("Could not recognize a subset ID from {}.".format(expression))
