"""notebooks/report.ipynb is the jupytext pair of notebooks/report.py and has no outputs."""

from pathlib import Path

import jupytext

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def test_the_notebook_and_its_script_hold_the_same_cells() -> None:
    script = jupytext.read(NOTEBOOKS / "report.py")
    notebook = jupytext.read(NOTEBOOKS / "report.ipynb")
    assert [(c.cell_type, c.source) for c in script.cells] == [
        (c.cell_type, c.source) for c in notebook.cells
    ]


def test_the_notebook_is_stripped() -> None:
    notebook = jupytext.read(NOTEBOOKS / "report.ipynb")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
