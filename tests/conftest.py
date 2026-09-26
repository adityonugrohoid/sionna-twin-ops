"""Test setup: the Mitsuba variant is selected before any test module imports Sionna RT
(repo rule 8), and every test runs with a pinned commit so provenance never needs git."""

from collections.abc import Iterator

import pytest

from sionna_twin_ops.backend import DEFAULT_VARIANT, select_variant

select_variant(DEFAULT_VARIANT)


@pytest.fixture(autouse=True)
def pinned_commit(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin TWIN_COMMIT so tests pass outside a git checkout."""
    monkeypatch.setenv("TWIN_COMMIT", "test-commit")
    yield
