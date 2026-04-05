# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ezconfy",
#     "torch",
#     "torchmetrics",
#     "torchvision",
# ]
# ///
from pathlib import Path
from typing import Any

import torch

from ezconfy import ConfigBuilder

# ---- Load config: the single source of truth for everything ----
example_dir = Path(__file__).parent
cfg: Any = ConfigBuilder.from_files(
    config_paths=example_dir / "config.yaml",
    schema_path=example_dir / "schema.yaml",
)

device = cfg.device
model = cfg.model.to(device)
metric = cfg.metric.to(device)

# ---- Train ----
for epoch in range(1, cfg.epochs + 1):
    model.train()
    for data, target in cfg.train_loader:
        data, target = data.to(device), target.to(device)
        cfg.optimizer.zero_grad()
        cfg.criterion(model(data), target).backward()
        cfg.optimizer.step()

    model.eval()
    metric.reset()
    with torch.inference_mode():
        for data, target in cfg.test_loader:
            data, target = data.to(device), target.to(device)
            metric.update(model(data), target)

    print(f"Epoch {epoch}/{cfg.epochs}  {metric.__class__.__name__.lower()}: {metric.compute():.2%}")
