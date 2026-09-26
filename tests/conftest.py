"""Select the Mitsuba variant before any test module imports Sionna RT (repo rule 8)."""

from sionna_twin_ops.backend import DEFAULT_VARIANT, select_variant

select_variant(DEFAULT_VARIANT)
