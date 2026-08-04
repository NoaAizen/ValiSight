#!/usr/bin/env python3
"""A single validation performed during initialization.

Every guard in the pipeline returns a Check rather than a bare bool, so the
assumption behind it stays visible: a failure report shows what was measured
against what was expected, instead of only saying that something failed.

That distinction is the point. The bugs this package was built around were all
assumptions buried in code — a hardcoded geoid offset, a plausibility range
that excluded the correct answer — where nothing in the output revealed that an
assumption was being made at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


class CheckFailed(RuntimeError):
    """Raised when a check that a stage cannot continue without does not pass."""

    def __init__(self, check: "Check") -> None:
        super().__init__(str(check))
        self.check = check


@dataclass(frozen=True)
class Check:
    """One validation, with the measurement and expectation that produced it."""

    name: str
    passed: bool
    detail: str
    measured: Optional[float] = None
    expected: Optional[Tuple[float, float]] = None
    unit: str = ""

    @classmethod
    def in_range(
        cls,
        name: str,
        value: float,
        low: float,
        high: float,
        unit: str = "",
        detail: str = "",
    ) -> "Check":
        """Check that a measured value falls within an inclusive range."""
        passed = low <= value <= high
        suffix = f" {unit}" if unit else ""
        return cls(
            name=name,
            passed=passed,
            detail=detail or (
                f"{value:.3f}{suffix} is within [{low:.3f}, {high:.3f}]{suffix}"
                if passed
                else f"{value:.3f}{suffix} is outside [{low:.3f}, {high:.3f}]{suffix}"
            ),
            measured=value,
            expected=(low, high),
            unit=unit,
        )

    @classmethod
    def that(cls, name: str, condition: bool, detail: str) -> "Check":
        """Check a condition that has no meaningful numeric measurement."""
        return cls(name=name, passed=condition, detail=detail)

    def raise_if_failed(self) -> "Check":
        """Return self, or raise CheckFailed when this check did not pass."""
        if not self.passed:
            raise CheckFailed(self)
        return self

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.detail}"
