from pathlib import Path
from pydantic import BaseModel, Field


class Dataset(BaseModel):
    root: Path = Field(...)
    name: str = Field(...)


class Model(BaseModel):
    type: str = Field(...)
    pretrained: bool = Field(...)


class ConfigModel(BaseModel):
    dataset: Dataset = Field(...)
    model: Model = Field(...)
