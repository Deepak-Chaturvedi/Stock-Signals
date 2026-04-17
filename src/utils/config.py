import yaml
from pathlib import Path


def load_config() -> dict:
    """
    Load config.yaml (ML + pipeline settings)
    """

    config_path = (
        Path(__file__)
        .resolve()
        .parent.parent.parent
        / "config"
        / "config.yaml"
    )

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)