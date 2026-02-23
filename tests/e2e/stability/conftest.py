# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Stability test helpers for L5 long-run testing.

This module provides fixtures and utilities for monitoring resource usage
(memory, VRAM, latency, throughput) over extended serving periods and
detecting drift that indicates leaks or degradation.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


@dataclass
class MetricsSample:
    """A single point-in-time snapshot of monitored metrics."""

    timestamp: float
    rss_memory_mb: float = 0.0
    vram_used_mb: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    error_count: int = 0


@dataclass
class StabilityReport:
    """Aggregated report for a stability test run."""

    test_name: str
    duration_seconds: float
    samples: list[MetricsSample] = field(default_factory=list)

    @property
    def baseline_window(self) -> list[MetricsSample]:
        """Return the first 10% of samples as the baseline window."""
        count = max(1, len(self.samples) // 10)
        return self.samples[:count]

    @property
    def tail_window(self) -> list[MetricsSample]:
        """Return the last 10% of samples as the tail window."""
        count = max(1, len(self.samples) // 10)
        return self.samples[-count:]

    def compute_drift(self, metric_name: str) -> float:
        """Compute percentage drift between baseline and tail windows.

        Returns:
            Percentage change from baseline mean to tail mean.
            Positive means the metric increased; negative means it decreased.
        """
        if len(self.samples) < 2:
            return 0.0

        baseline_vals = [getattr(s, metric_name) for s in self.baseline_window]
        tail_vals = [getattr(s, metric_name) for s in self.tail_window]

        baseline_mean = sum(baseline_vals) / len(baseline_vals)
        tail_mean = sum(tail_vals) / len(tail_vals)

        if baseline_mean == 0:
            return 0.0
        return ((tail_mean - baseline_mean) / baseline_mean) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a dictionary for JSON export."""
        return {
            "test_name": self.test_name,
            "duration_seconds": self.duration_seconds,
            "num_samples": len(self.samples),
            "drifts": {
                "rss_memory_mb": self.compute_drift("rss_memory_mb"),
                "vram_used_mb": self.compute_drift("vram_used_mb"),
                "p99_latency_ms": self.compute_drift("p99_latency_ms"),
                "throughput_rps": self.compute_drift("throughput_rps"),
                "error_count": self.compute_drift("error_count"),
            },
        }


def load_stability_configs(config_path: str | None = None) -> list[dict[str, Any]]:
    """Load stability test configurations from a JSON file.

    Args:
        config_path: Path to the JSON config. Defaults to weekly.json
                     in the same directory as this module.

    Returns:
        List of stability test configuration dicts.
    """
    if config_path is None:
        config_path = str(Path(__file__).parent / "weekly.json")

    abs_path = Path(config_path).resolve()
    with open(abs_path, encoding="utf-8") as f:
        return json.load(f)


def assert_stability_thresholds(
    report: StabilityReport,
    thresholds: dict[str, float],
) -> None:
    """Assert that no metric drifted beyond the configured thresholds.

    Args:
        report: The completed stability report.
        thresholds: Mapping of threshold keys to maximum allowed drift %.
                    Keys follow the pattern ``max_{metric}_drift_pct``.
    """
    drift_map = {
        "max_memory_drift_pct": "rss_memory_mb",
        "max_vram_drift_pct": "vram_used_mb",
        "max_p99_latency_drift_pct": "p99_latency_ms",
        "max_throughput_drop_pct": "throughput_rps",
        "max_error_rate_pct": "error_count",
    }

    for threshold_key, metric_name in drift_map.items():
        max_drift = thresholds.get(threshold_key)
        if max_drift is None:
            continue

        actual_drift = report.compute_drift(metric_name)

        if metric_name == "throughput_rps":
            assert actual_drift >= -max_drift, f"Throughput dropped {abs(actual_drift):.1f}% (threshold: {max_drift}%)"
        else:
            assert actual_drift <= max_drift, f"{metric_name} drifted +{actual_drift:.1f}% (threshold: {max_drift}%)"


def collect_process_memory_mb() -> float:
    """Return the RSS memory of the current process in MiB."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def collect_vram_used_mb(device_index: int = 0) -> float:
    """Return VRAM usage for the given device in MiB."""
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            return (total_bytes - free_bytes) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


@pytest.fixture(scope="module")
def stability_configs() -> list[dict[str, Any]]:
    """Fixture that loads stability test configurations."""
    return load_stability_configs()
