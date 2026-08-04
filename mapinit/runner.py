#!/usr/bin/env python3
"""Runs the initialization stages in order and collects their results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from .check import Check
from .context import InitContext
from .stage import InitStage, StageResult, StageStatus


@dataclass
class InitReport:
    """Outcome of a full initialization run."""

    results: List[StageResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.results) and all(r.ok for r in self.results)

    @property
    def checks(self) -> List[Check]:
        return [check for result in self.results for check in result.checks]

    @property
    def failed_checks(self) -> List[Check]:
        return [check for check in self.checks if not check.passed]

    def stage(self, name: str) -> StageResult:
        for result in self.results:
            if result.stage == name:
                return result
        raise KeyError(f"Stage {name!r} did not run. Ran: {[r.stage for r in self.results]}")

    def summary(self) -> str:
        passed = sum(1 for c in self.checks if c.passed)
        parts = [f"{passed}/{len(self.checks)} checks passed"]

        # A stage can fail by raising before it records any check, so counting
        # checks alone would report "all passed" on a failed run
        failed_stages = [r.stage for r in self.results if r.failed]
        if failed_stages:
            parts.append(f"failed at {', '.join(failed_stages)}")
        skipped = [r.stage for r in self.results if r.status is StageStatus.SKIPPED]
        if skipped:
            parts.append(f"skipped {', '.join(skipped)}")

        header = "Initialization succeeded" if self.ok else "Initialization FAILED"
        lines = [f"{header} ({'; '.join(parts)})", ""]
        lines.extend(result.summary() for result in self.results)
        return "\n".join(lines)


class InitializationPipeline:
    """Ordered collection of stages that initializes position against known maps."""

    def __init__(self, stages: Sequence[InitStage]) -> None:
        self.stages = list(stages)

    @classmethod
    def default(cls) -> "InitializationPipeline":
        """The standard stage order.

        The geoid guard runs first deliberately. It is the cheapest check and
        the one that invalidates everything downstream: surveyed target heights
        and rig heights must share a vertical datum before calibration means
        anything, and GLO-30 elevations are themselves referenced to EGM2008.
        """
        # Imported here so that importing the runner does not pull in pyproj
        from .stages.calibration import CalibrationStage
        from .stages.geoid import GeoidStage
        from .stages.priors import PriorsStage

        return cls([GeoidStage(), PriorsStage(), CalibrationStage()])

    def run(self, ctx: InitContext, fail_fast: bool = True) -> InitReport:
        """Execute every stage in order.

        With ``fail_fast`` the run stops at the first failing stage, which is
        what initialization wants. With ``fail_fast=False`` every stage runs and
        the report covers all of them, which is what diagnosing a broken setup
        wants — one run surfaces every problem instead of one per fix.
        """
        report = InitReport()

        for stage in self.stages:
            result = stage.run(ctx)
            report.results.append(result)
            ctx.results[stage.name] = result

            if result.failed and fail_fast:
                break

        return report
