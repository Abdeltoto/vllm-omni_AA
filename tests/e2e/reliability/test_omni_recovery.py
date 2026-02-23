# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
L5 Reliability tests -- recovery / chaos scenarios for omni models.

These tests inject faults into a running vLLM-Omni serving instance and
verify that the system recovers within an acceptable time budget.

Fault categories:
  (a) Process kill & restart  -- validates auto-recovery / graceful restart
  (b) Abnormal input          -- validates error handling without crash
  (c) Resource pressure (OOM) -- validates graceful degradation

Run:
    pytest -s -v tests/e2e/reliability/test_omni_recovery.py
"""

import os
import time

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.e2e.reliability.conftest import (
    FaultType,
    RecoveryResult,
    assert_graceful_recovery,
    wait_for_healthy,
)

pytestmark = [pytest.mark.slow]


# ---------------------------------------------------------------------------
# Abnormal-input recovery
# ---------------------------------------------------------------------------
@pytest.mark.core_model
@pytest.mark.omni
class TestAbnormalInputRecovery:
    """Verify the server stays healthy after receiving malformed requests."""

    @staticmethod
    def _send_malformed_request(client: object, model: str) -> None:
        """Send a deliberately malformed chat-completion request."""
        import openai

        assert isinstance(client, openai.OpenAI)
        try:
            client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "unknown_modality", "data": "garbage"},
                        ],
                    }
                ],
                stream=False,
            )
        except Exception:
            pass

    @staticmethod
    def _server_is_healthy(client: object, model: str) -> bool:
        """Return True if a simple text request succeeds."""
        import openai

        assert isinstance(client, openai.OpenAI)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say hello."}],
                max_tokens=8,
                stream=False,
            )
            return resp.choices[0].message.content is not None
        except Exception:
            return False

    def test_abnormal_input_recovery(self, client: object, omni_server: object) -> None:
        """Inject malformed input and assert the server remains healthy.

        Deploy Setting: default yaml
        Fault: unknown_modality payload
        Expectation: server returns error but stays alive for subsequent
                     valid requests.
        """
        model = omni_server.model  # type: ignore[attr-defined]
        injection_time = time.monotonic()

        self._send_malformed_request(client, model)

        healthy = wait_for_healthy(
            lambda: self._server_is_healthy(client, model),
            timeout_seconds=30.0,
        )
        recovery_time = time.monotonic() if healthy else None

        result = RecoveryResult(
            fault_type=FaultType.ABNORMAL_INPUT,
            injection_time=injection_time,
            recovery_time=recovery_time,
            recovered=healthy,
            error_message=None if healthy else "Server unresponsive after malformed input",
        )
        assert_graceful_recovery(result, max_recovery_seconds=30.0)
