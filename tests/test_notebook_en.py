"""Validation tests for transcribe_colab_en.ipynb."""

from __future__ import annotations

from pathlib import Path

import pytest


nbformat = pytest.importorskip("nbformat")

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "transcribe_colab_en.ipynb"


def test_notebook_en_is_valid_nbformat() -> None:
    assert NOTEBOOK_PATH.is_file()
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells) >= 5


def test_notebook_en_contains_required_config_and_calls_src() -> None:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_sources = [cell["source"] for cell in nb.cells if cell["cell_type"] == "code"]
    joined = "\n".join(code_sources)

    for var_name in ("INPUT_DIR", "LANGUAGE", "MODEL", "OVERWRITE"):
        assert var_name in code_sources[0], f"First code cell must define {var_name}"

    assert 'LANGUAGE = "en"' in code_sources[0]
    assert "files.upload" in joined
    assert "faster-whisper" in joined
    assert "from src.transcribe import transcribe_directory" in joined
