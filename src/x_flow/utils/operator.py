from typing import Callable

from pydantic import BaseModel, Field, field_validator


class Operator(BaseModel):
    operator: str = Field(...)

    @field_validator("operator")
    def normalize_operation(cls, v):
        operation_mapping = {
            "<": ["lt", "less than", "less"],
            ">": ["gt", "greater than", "greater"],
            "<=": ["lte", "less than or equal to", "less equal"],
            ">=": ["gte", "greater than or equal to", "greater equal"],
            "==": ["eq", "equal to", "equals", "equal"],
            "!=": ["ne", "not equal to", "not equals", "not equal"],
        }

        for op, aliases in operation_mapping.items():
            if v.lower() in aliases + [op]:
                return op

        raise ValueError(
            f"""Invalid operation: {v}
            allowed Values:
            {operation_mapping}
        """
        )

    def apply_operation(self, threshold: float) -> Callable[[float], bool]:
        return {
            ">": lambda x: x > threshold,
            "<": lambda x: x < threshold,
            ">=": lambda x: x >= threshold,
            "<=": lambda x: x <= threshold,
            "==": lambda x: x == threshold,
            "!=": lambda x: x != threshold,
        }[self.operator]
