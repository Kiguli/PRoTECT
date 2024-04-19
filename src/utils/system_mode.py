from enum import Enum


class SystemMode(str, Enum):
    """
    Enumeration for system configuration types.

    Possible values:
    - SystemMode.DT_DS: "discrete_deterministic"
    - SystemMode.DT_SS: "discrete_stochastic"
    - SystemMode.CT_DS: "continuous_deterministic"
    - SystemMode.CT_SS: "continuous_stochastic"
    """

    DT_DS = "discrete_deterministic"
    DT_SS = "discrete_stochastic"
    CT_DS = "continuous_deterministic"
    CT_SS = "continuous_stochastic"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.value.lower())

    def is_stochastic(self) -> bool:
        return self == SystemMode.DT_SS or self == SystemMode.CT_SS

    def is_discrete(self) -> bool:
        return self == SystemMode.DT_DS or self == SystemMode.DT_SS
