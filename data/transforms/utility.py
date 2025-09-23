from monai.data import MetaTensor

from data.transforms import SerializableTransform


class Rename(SerializableTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, data: MetaTensor) -> MetaTensor:
        return data
    
    @classmethod
    def json_identifier(cls) -> str:
        return "rename"

    # noinspection PyMethodMayBeStatic
    def to_json(self):
        return {}
