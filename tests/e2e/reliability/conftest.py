# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Reliability / chaos-test helpers for L5 recovery testing.

This module provides fixtures and utilities for simulating fault scenarios
(process kill, OOM simulation, abnormal inputs, network disruption) and
verifying that the serving system recovers gracefully.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pytest


class FaultType(Enum):
    """Catalogue of injectable fault scenarios."""

    PROCESS_KILL = "process_kill"
    OOM_SIMULATION = "oom_simulation"
    ABNORMAL_INPUT = "abnormal_input"
    NETWORK_DISRUPTION = "network_disruption"
    WORKER_RESTART = "worker_restart"


@dataclass
class RecoveryResult:
    """Outcome of a single fault-injection + recovery cycle."""

    fault_type: FaultType
    injection_time: float
    recovery_time: float | None
    recovered: bool
    error_message: str | None = None

    @property
    def recovery_duration_seconds(self) -> float | None:
        """Wall-clock seconds between fault injection and full recovery."""
        if self.recovery_time is None:
            return None
        return self.recovery_time - self.injection_time

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON reporting."""
        return {
            "fault_type": self.fault_type.value,
            "recovered": self.recovered,
            "recovery_duration_s": self.recovery_duration_seconds,
            "error_message": self.error_message,
        }


def wait_for_healthy(
    health_check_fn: Any,
    timeout_seconds: float = 120.0,
    poll_interval: float = 2.0,
) -> bool:
    """Poll a health-check callable until it returns ``True`` or timeout.

    Args:
        health_check_fn: Zero-arg callable returning ``True`` when service
                         is healthy.
        timeout_seconds: Maximum wait time.
        poll_interval: Seconds between polls.

    Returns:
        ``True`` if the service became healthy within the timeout.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            if health_check_fn():
                return True
        except Exception:
            pass
        time.sleep(poll_interval)
    return False


def assert_graceful_recovery(
    result: RecoveryResult,
    max_recovery_seconds: float = 120.0,
) -> None:
    """Assert that the system recovered within the allowed time budget.

    Args:
        result: The recovery outcome to validate.
        max_recovery_seconds: Upper bound for acceptable recovery time.
    """
    assert result.recovered, f"System did not recover from {result.fault_type.value}: {result.error_message}"
    duration = result.recovery_duration_seconds
    assert duration is not None
    assert duration <= max_recovery_seconds, f"Recovery from {result.fault_type.value} took {duration:.1f}s (budget: {max_recovery_seconds}s)"


@pytest.fixture
def fault_types() -> list[FaultType]:
    """Fixture exposing the available fault types for parametrized tests."""
    return list(FaultType)
