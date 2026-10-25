import pytest

from nl_to_sql import config


@pytest.fixture(autouse=True)
def _no_description_generation(monkeypatch):
    """Tests use scripted LLMs; never let description generation consume their replies."""
    monkeypatch.setattr(config, "AUTO_DESCRIBE", False)
