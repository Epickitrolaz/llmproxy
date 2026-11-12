import shutil
from pathlib import Path
import pytest

from llmproxy.core.manager import plugin_manager

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
PLUGINS_DIR = Path(__file__).resolve().parent.parent / "plugins"

# Ensure templates exist in plugins for test execution and imports
if TEMPLATES_DIR.exists():
    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
    for tpl_file in TEMPLATES_DIR.glob("*.py"):
        target = PLUGINS_DIR / tpl_file.name
        if not target.exists():
            shutil.copy(tpl_file, target)


@pytest.fixture(autouse=True)
def admin_token_env(monkeypatch):
    """Give the dashboard API tests a known admin token."""
    monkeypatch.setenv("LLMPROXY_ADMIN_TOKEN", "test-admin-token")


@pytest.fixture(autouse=True)
def prepare_test_environment(request):
    """
    Ensure built-in templates are populated into plugin_manager for functional
    tests that expect default providers, while preserving isolation for dashboard tests.
    """
    if "test_dashboard" in str(request.node.fspath):
        yield
        return

    plugin_manager.discover_and_load_all()
    yield
