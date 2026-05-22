"""PolicyAction — the Brain's Command Interface (Policy Mapping).

Maps 13 discrete UI/MQTT commands to ManifoldBrain methods, providing a
clean, extensible API contract between the frontend (CoC dashboard) and the
spatial AI backend.

The ``PolicyAction`` IntEnum ensures frontend and backend stay in sync.
``PolicyCommandPayload`` is the Pydantic model that arrives on the MQTT
``cmd`` topic.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)


class PolicyAction(IntEnum):
    """Discrete policy actions the UI can dispatch to the Brain.

    The numeric codes are stable — new actions must be appended, never
    reordered — so that MQTT payloads from older frontends remain valid.
    """

    DEPLOY_AGENT = 1
    GATHER_DATA = 2
    RECALIBRATE = 3
    PATROL = 4
    MAINTENANCE = 5
    SCAN_WORLD = 6
    INTERACT_OBJECT = 7
    DEFEND_ZONE = 8
    ESCALATE = 9
    RETURN_HOME = 10
    FORM_COALITION = 11
    RESEARCH = 12
    EMERGENCY_STOP = 13


class PolicyCommandPayload(BaseModel):
    """Pydantic model for inbound command payloads on the MQTT ``cmd`` topic.

    Attributes
    ----------
    action_code:
        Integer matching a :class:`PolicyAction` member.
    params:
        Action-specific parameters (e.g. target coordinates, zone id).
    request_id:
        Optional client-side correlation id for response tracking.
    """

    action_code: int
    params: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None
