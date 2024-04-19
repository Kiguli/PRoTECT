from enum import Enum


class NoiseType(str, Enum):
    """
    Enumeration for noise types.

    Possible values:
    - NoiseType.normal: "Normal"
    - NoiseType.exponential: "Exponential"
    - NoiseType.uniform: "Uniform"
    """

    normal = "Normal"
    exponential = "Exponential"
    uniform = "Uniform"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.value.lower())
