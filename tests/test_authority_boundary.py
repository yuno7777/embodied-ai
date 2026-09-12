from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_production_python_control_plane_does_not_import_legacy_simulator():
    production_modules = (ROOT / "embodied_ai").glob("*.py")
    offenders = [
        path.name
        for path in production_modules
        if path.name != "engine.py" and "from .engine import" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_quarantine_documentation_names_replaced_systems():
    document = (ROOT / "docs" / "legacy-systems.md").read_text(encoding="utf-8")
    assert "Rust `sim-core`" in document
    assert "embodied_ai.engine" in document
    assert "frontend/observer" in document
