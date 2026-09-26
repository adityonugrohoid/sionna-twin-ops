"""The one place Sionna RT is imported; refuses to import before a variant is selected."""

import mitsuba as mi

from sionna_twin_ops.backend import active_variant

active_variant()  # raises if no variant has been selected

import sionna.rt as rt  # noqa: E402  (must follow the variant check)

__all__ = ["mi", "rt"]
