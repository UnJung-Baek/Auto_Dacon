from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_auto_dacon():
    module_path = REPO_ROOT / "auto_dacon.py"
    package_path = REPO_ROOT / "src" / "auto_dacon" / "__init__.py"

    package_spec = importlib.util.spec_from_file_location(
        "auto_dacon",
        package_path,
        submodule_search_locations=[str(package_path.parent)],
    )
    package_module = importlib.util.module_from_spec(package_spec)
    assert package_spec.loader is not None
    package_spec.loader.exec_module(package_module)
    sys.modules["auto_dacon"] = package_module

    spec = importlib.util.spec_from_file_location("auto_dacon_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_research_model_defaults_are_role_specific_and_static() -> None:
    module = _load_auto_dacon()

    assert module.DEFAULT_RESEARCH_ANALYST_MODELS == (
        "deepseek/deepseek-v4-pro",
        "google/gemini-3.5-flash",
        "z-ai/glm-5.1",
    )
    assert module.DEFAULT_RESEARCH_HYPOTHESIS_MODELS == (
        "anthropic/claude-sonnet-4.6",
        "deepseek/deepseek-v4-pro",
        "moonshotai/kimi-k2.7-code",
    )
    assert module.DEFAULT_RESEARCH_CRITIC_MODELS == (
        "anthropic/claude-sonnet-4.6",
        "deepseek/deepseek-v4-pro",
        "google/gemini-3.5-flash",
    )

    assert len(module.DEFAULT_RESEARCH_ANALYST_MODELS) == 3
    assert len(module.DEFAULT_RESEARCH_HYPOTHESIS_MODELS) == 3
    assert len(module.DEFAULT_RESEARCH_CRITIC_MODELS) == 3

    assert "openai/gpt-5.5" not in module.DEFAULT_RESEARCH_ANALYST_MODELS
    assert "openai/gpt-5.5" not in module.DEFAULT_RESEARCH_HYPOTHESIS_MODELS
    assert "openai/gpt-5.5" not in module.DEFAULT_RESEARCH_CRITIC_MODELS

    assert module.DEFAULT_RESEARCH_SELECTOR_MODEL == "anthropic/claude-sonnet-4.6"
    assert module.DEFAULT_RESEARCH_WARM_START_MODEL == "anthropic/claude-sonnet-4.6"
