"""Mitsuba variant selection: explicit, recorded, never auto-detected (repo rule 8).

The variant must be selected with `select_variant` before `sionna_twin_ops.sionna_rt`
(and anything that imports it) is imported. Importing Sionna RT with no variant set would
let it pick one on its own, so that import refuses to run until a variant is chosen.
"""

import mitsuba as mi

DEFAULT_VARIANT = "llvm_ad_mono_polarized"


def select_variant(variant: str) -> None:
    """Set the Mitsuba variant for this process.

    Args:
        variant: Mitsuba variant name, e.g. DEFAULT_VARIANT.

    Raises:
        ValueError: If the variant is not installed.
        RuntimeError: If a different variant is already set in this process.
    """
    if variant not in mi.variants():
        raise ValueError(f"Mitsuba variant {variant!r} is not installed; have {mi.variants()}")
    current = mi.variant()
    if current is None:
        mi.set_variant(variant)
    elif current != variant:
        raise RuntimeError(
            f"Mitsuba variant {current!r} is already set; cannot switch to {variant!r}"
        )


def active_variant() -> str:
    """The selected Mitsuba variant.

    Returns:
        The variant name.

    Raises:
        RuntimeError: If no variant has been selected.
    """
    variant: str | None = mi.variant()
    if variant is None:
        raise RuntimeError("no Mitsuba variant selected: call backend.select_variant first")
    return variant
