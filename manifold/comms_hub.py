"""manifold/comms_hub.py — CommHub multi-channel dispatcher.

Routes escalation notifications to the right people via the right
channel (push, email, Slack, SMS, webhook, or world dashboard),
using vocabulary appropriate for the recipient's user type.

External channels (EMAIL, SLACK, SMS) require env vars:
  MANIFOLD_SMTP_HOST — SMTP server hostname for email
  MANIFOLD_SMS_URL   — Twilio-compatible REST endpoint for SMS

If an env var is missing the channel is skipped with a warning.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Channel enum + config
# ---------------------------------------------------------------------------

class CommChannel(str, Enum):
    PUSH = "push"
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    WEBHOOK = "webhook"
    WORLD_DASHBOARD = "world_dashboard"


@dataclass
class ChannelConfig:
    """Configuration for a single communication channel.

    Parameters
    ----------
    channel:
        Which channel to use.
    address:
        Phone number, email address, Slack webhook URL, or HTTP endpoint.
    min_risk:
        Only send when ``escalation["risk_score"] >= min_risk``.
    max_risk:
        Only send when ``escalation["risk_score"] <= max_risk``.
    user_type:
        Vocabulary level for the message (developer/executive/…).
    """
    channel: CommChannel
    address: str
    min_risk: float = 0.0
    max_risk: float = 1.0
    user_type: str = "executive"

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel.value,
            "address": self.address,
            "min_risk": self.min_risk,
            "max_risk": self.max_risk,
            "user_type": self.user_type,
        }


# ---------------------------------------------------------------------------
# CommHub
# ---------------------------------------------------------------------------

class CommHub:
    """Dispatches escalation notifications across multiple channels.

    Channels are registered per owner_id.  When ``dispatch()`` is called
    for an owner, every channel whose risk window covers the escalation
    risk_score receives an appropriately formatted message.
    """

    def __init__(self) -> None:
        # owner_id → list[ChannelConfig]
        self._channels: dict[str, list[ChannelConfig]] = {}

    # ------------------------------------------------------------------

    def register_channel(self, owner_id: str, config: ChannelConfig) -> None:
        """Register or replace a channel config for owner_id."""
        existing = self._channels.setdefault(owner_id, [])
        # Replace existing config for same channel type
        self._channels[owner_id] = [
            c for c in existing if c.channel != config.channel
        ]
        self._channels[owner_id].append(config)

    def remove_channel(self, owner_id: str, channel: CommChannel) -> bool:
        """Remove a channel for owner_id.  Returns True if found."""
        before = len(self._channels.get(owner_id, []))
        self._channels[owner_id] = [
            c for c in self._channels.get(owner_id, [])
            if c.channel != channel
        ]
        return len(self._channels.get(owner_id, [])) < before

    def channels_for(self, owner_id: str) -> list[ChannelConfig]:
        """Return all registered channels for owner_id."""
        return list(self._channels.get(owner_id, []))

    def dispatch(self, owner_id: str, escalation: dict) -> list[str]:
        """Send escalation notifications to all matching channels.

        Parameters
        ----------
        owner_id:
            The owner whose channels to consult.
        escalation:
            Dict with at least ``risk_score`` and ``action`` keys.

        Returns
        -------
        list[str]
            Names of channels that were successfully dispatched.
        """
        risk = float(escalation.get("risk_score", 0.5))
        dispatched: list[str] = []

        for config in self._channels.get(owner_id, []):
            if not (config.min_risk <= risk <= config.max_risk):
                continue
            try:
                message = self._build_message(escalation, config.user_type)
                ok = self._send(config, message, escalation)
                if ok:
                    dispatched.append(config.channel.value)
            except Exception as exc:  # noqa: BLE001
                log.warning("CommHub: channel %s failed: %s", config.channel, exc)

        return dispatched

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_message(self, escalation: dict, user_type: str) -> str:
        """Delegate message building to GovernanceReporter."""
        try:
            from manifold.governance_reporter import GovernanceReporter  # noqa: PLC0415
            reporter = GovernanceReporter()
            return reporter.generate_escalation_message(escalation, user_type)
        except Exception:  # noqa: BLE001
            action = escalation.get("action", "unknown")
            risk = escalation.get("risk_score", 0)
            return f"MANIFOLD escalation: action={action}, risk={risk:.2f}"

    # ------------------------------------------------------------------
    # Channel adapters
    # ------------------------------------------------------------------

    def _send(self, config: ChannelConfig, message: str, escalation: dict) -> bool:
        """Dispatch to the appropriate adapter.  Returns True on success."""
        ch = config.channel
        if ch == CommChannel.PUSH:
            return self._send_push(config, message)
        if ch == CommChannel.EMAIL:
            return self._send_email(config, message)
        if ch == CommChannel.SLACK:
            return self._send_slack(config, message)
        if ch == CommChannel.SMS:
            return self._send_sms(config, message)
        if ch == CommChannel.WEBHOOK:
            return self._send_webhook(config, message, escalation)
        if ch == CommChannel.WORLD_DASHBOARD:
            return self._send_world_dashboard(escalation)
        return False

    def _send_push(self, config: ChannelConfig, message: str) -> bool:
        try:
            from manifold.remote import MobileAlertGateway  # noqa: PLC0415
            gw = MobileAlertGateway()
            result = gw.deliver_sync(message, urgency="high", agent_id=config.address)
            return bool(result.get("delivered", False))
        except Exception as exc:  # noqa: BLE001
            log.warning("CommHub PUSH failed: %s", exc)
            return False

    def _send_email(self, config: ChannelConfig, message: str) -> bool:
        smtp_host = os.environ.get("MANIFOLD_SMTP_HOST", "")
        if not smtp_host:
            log.warning("CommHub EMAIL skipped: MANIFOLD_SMTP_HOST not set")
            return False
        try:
            smtp_port = int(os.environ.get("MANIFOLD_SMTP_PORT", "25"))
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as smtp:
                from_addr = os.environ.get("MANIFOLD_SMTP_FROM", "manifold@localhost")
                body = (
                    f"From: {from_addr}\r\nTo: {config.address}\r\n"
                    f"Subject: MANIFOLD Escalation\r\n\r\n{message}"
                )
                smtp.sendmail(from_addr, [config.address], body)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("CommHub EMAIL failed: %s", exc)
            return False

    def _send_slack(self, config: ChannelConfig, message: str) -> bool:
        if not config.address:
            log.warning("CommHub SLACK skipped: no webhook URL configured")
            return False
        try:
            payload = json.dumps({"text": message}).encode()
            req = urllib.request.Request(
                config.address,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10):  # noqa: S310
                pass
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("CommHub SLACK failed: %s", exc)
            return False

    def _send_sms(self, config: ChannelConfig, message: str) -> bool:
        sms_url = os.environ.get("MANIFOLD_SMS_URL", "")
        if not sms_url:
            log.warning("CommHub SMS skipped: MANIFOLD_SMS_URL not set")
            return False
        try:
            payload = json.dumps({"to": config.address, "body": message}).encode()
            req = urllib.request.Request(
                sms_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10):  # noqa: S310
                pass
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("CommHub SMS failed: %s", exc)
            return False

    def _send_webhook(
        self, config: ChannelConfig, message: str, escalation: dict
    ) -> bool:
        if not config.address:
            log.warning("CommHub WEBHOOK skipped: no address configured")
            return False
        try:
            payload = json.dumps({
                "message": message,
                "escalation": escalation,
            }).encode()
            req = urllib.request.Request(
                config.address,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10):  # noqa: S310
                pass
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("CommHub WEBHOOK failed: %s", exc)
            return False

    def _send_world_dashboard(self, escalation: dict) -> bool:
        try:
            from manifold.cell_update_bus import get_bus, CellUpdate, CellCoord  # noqa: PLC0415
            bus = get_bus()
            bus.publish(CellUpdate(
                coord=CellCoord(x=0, y=0, z=0),
                r_delta=0.0,
                n_delta=0.0,
                a_delta=0.0,
                c_delta=0.0,
                source="comms_hub:escalation",
                ttl=60,
                metadata={"event_type": "escalation", **escalation},
            ))
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("CommHub WORLD_DASHBOARD failed: %s", exc)
            return False

    def summary(self) -> dict[str, Any]:
        return {
            "owners": len(self._channels),
            "total_channels": sum(len(v) for v in self._channels.values()),
            "channels": {
                owner: [c.to_dict() for c in configs]
                for owner, configs in self._channels.items()
            },
        }
