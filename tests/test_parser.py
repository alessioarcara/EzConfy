import pytest
from pydantic import ValidationError

from easyconfig.parser import SchemaParser

config = """
wandb_run_name: str
model:
  hidden_dims: list[int]
training:
  batch_size: int = 32
  num_epochs: int = 10
"""


def test_should_generate_pydantic_model_from_yaml_string():
    model = SchemaParser.parse(config)
    input_data = {
        "training": {"batch_size": 64},
        "model": {"hidden_dims": [128, 64]},
        "wandb_run_name": "test_run",
    }

    instance = model.validate(input_data)

    assert instance.training.batch_size == 64
    assert instance.training.num_epochs == 10
    assert instance.model.hidden_dims == [128, 64]
    assert instance.wandb_run_name == "test_run"


@pytest.mark.parametrize(
    "input_data",
    [
        pytest.param(
            {
                "training": {"batch_size": 64},
                "model": {"hidden_dims": [128, 64]},
            },
            id="missing_required_field",
        ),
        pytest.param(
            {
                "training": {"batch_size": "not_a_number"},
                "model": {"hidden_dims": [128, 64]},
                "wandb_run_name": "test_run",
            },
            id="invalid_batch_size_type",
        ),
    ],
)
def test_should_raise_validation_error(input_data):
    model = SchemaParser.parse(config)

    with pytest.raises(ValidationError):
        model.validate(input_data)
