#!/usr/bin/env python3
"""The contract every initialization stage implements.

Subclasses implement ``execute`` and return the checks they ran plus whatever
outputs later stages need. The base class turns that into a StageResult,
capturing exceptions so one failing stage cannot take down the runner and so a
diagnostic run can report every stage rather than only the first failure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple  # noqa: F401

from .check import Check, CheckFailed

if TYPE_CHECKING:
    from .context import InitContext


class StageStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Outcome of one stage: its checks, its outputs, and why it failed."""

    stage: str
    status: StageStatus
    checks: List[Check] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[BaseException] = None

    @property
    def ok(self) -> bool:
        """True when the stage did not fail. A skipped stage does not fail a run."""
        return self.status is not StageStatus.FAILED

    @property
    def failed(self) -> bool:
        return self.status is StageStatus.FAILED

    @property
    def failed_checks(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        lines = [f"{self.stage}: {self.status.value}"]
        lines.extend(f"    {check}" for check in self.checks)
        if self.error is not None:
            # Indent multi-line errors so they stay visually inside the stage
            error_text = str(self.error).replace("\n", "\n      ")
            lines.append(f"    {type(self.error).__name__}: {error_text}")
        return "\n".join(lines)


class InitStage(ABC):
    """Base class for one step of initialization.

    Adding a stage means subclassing this and registering it in
    ``InitializationPipeline.default()``; nothing else in the package changes.
    """

    #: Stable identifier used in reports and to look up this stage's outputs.
    name: ClassVar[str] = "stage"

    @abstractmethod
    def execute(self, ctx: "InitContext") -> Tuple[List[Check], Dict[str, Any]]:
        """Perform the stage's work.

        Returns the checks that were run and the outputs to publish for later
        stages. Raise for a hard failure that leaves nothing worth reporting.
        """

    def skip_reason(self, ctx: "InitContext") -> Optional[str]:
        """Return why this stage has nothing to do, or None to execute it.

        A stage that is not configured yet is neither a pass nor a failure, and
        reporting it as either would be misleading — one hides unfinished setup,
        the other blocks initialization over work that was never requested.
        """
        return None

    def run(self, ctx: "InitContext") -> StageResult:
        """Execute the stage and wrap the outcome, including any exception."""
        reason = self.skip_reason(ctx)
        if reason is not None:
            return StageResult(
                self.name,
                StageStatus.SKIPPED,
                [Check.that(f"{self.name}.configured", True, f"skipped: {reason}")],
            )

        try:
            checks, data = self.execute(ctx)
        except BaseException as exc:  # noqa: BLE001 - recorded, then re-surfaced by the runner
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            # Keep the failing check in the report rather than only its message
            checks = [exc.check] if isinstance(exc, CheckFailed) else []
            return StageResult(self.name, StageStatus.FAILED, checks, {}, error=exc)

        status = StageStatus.OK if all(c.passed for c in checks) else StageStatus.FAILED
        return StageResult(self.name, status, checks, data)
