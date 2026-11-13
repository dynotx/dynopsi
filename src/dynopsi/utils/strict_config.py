from pydantic import BaseModel, ConfigDict


class StrictConfig(BaseModel):
    """BaseModel that rejects unknown keys at validation (``extra="forbid"``)."""

    model_config = ConfigDict(extra="forbid")
