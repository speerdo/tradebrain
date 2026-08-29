"""
Notifier — Discord webhook fallback.

If Burt (Discord bot) is not running, falls back to webhook embeds.
"""

from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

import config


@dataclass
class Notification:
    title: str
    description: str
    color: int
    fields: list[dict] | None = None


class Notifier:
    """Sends notifications via Discord webhook or delegates to Burt."""

    COLORS = {
        "long": 0x00FF88,
        "short": 0xFF4455,
        "profit": 0x00FF88,
        "loss": 0xFF4455,
        "warning": 0xFFAA00,
        "circuit": 0xFF0000,
        "info": 0x00AAFF,
    }

    def __init__(self, burt: Any = None):
        self.cfg = config.get_config()
        self.burt = burt
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(15.0))
        self._has_webhook = bool(self.cfg.discord_webhook_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def notify_trade_opened(self, symbol: str, direction: str,
                                   entry_price: float, size_usdc: float,
                                   leverage: int, stop_loss: float,
                                   take_profit: float) -> None:
        """Notify that a trade was opened."""
        # Burt first; fall through to the webhook if he could not post.
        # The old code built a stub signal carrying ONLY `direction`, so entry
        # price, size, leverage, stop and target were discarded and the Discord
        # message was "Opened long <id>." with no numbers.
        if self.burt and await self.burt.notify_trade_opened(
            symbol, direction, entry_price, size_usdc, leverage,
            stop_loss, take_profit,
            "PAPER" if self.cfg.paper_trading else "LIVE",
        ):
            return

        color = self.COLORS.get(direction, self.COLORS["info"])
        notif = Notification(
            title=f"{'🟢' if direction == 'long' else '🔴'} Trade Opened: {symbol}",
            description=f"{direction.upper()} @ {entry_price:.2f}",
            color=color,
            fields=[
                {"name": "Size", "value": f"${size_usdc:.2f}", "inline": True},
                {"name": "Leverage", "value": f"{leverage}x", "inline": True},
                {"name": "Stop Loss", "value": f"{stop_loss:.2f}", "inline": True},
                {"name": "Take Profit", "value": f"{take_profit:.2f}", "inline": True},
            ],
        )
        await self._send_webhook(notif)

    async def notify_trade_closed(self, symbol: str, direction: str,
                                   entry_price: float, exit_price: float,
                                   pnl_usdc: float) -> None:
        """Notify that a trade was closed."""
        if self.burt and await self.burt.notify_trade_closed(
            symbol, direction, entry_price, exit_price, pnl_usdc
        ):
            return

        is_profit = pnl_usdc >= 0
        color = self.COLORS["profit"] if is_profit else self.COLORS["loss"]
        notif = Notification(
            title=f"{'🟢' if is_profit else '🔴'} Trade Closed: {symbol}",
            description=f"P&L: ${pnl_usdc:+.2f}",
            color=color,
            fields=[
                {"name": "Entry", "value": f"{entry_price:.2f}", "inline": True},
                {"name": "Exit", "value": f"{exit_price:.2f}", "inline": True},
                {"name": "Direction", "value": direction.upper(), "inline": True},
            ],
        )
        await self._send_webhook(notif)

    async def notify_circuit_breaker(self, daily_loss: float, limit: float) -> None:
        """Notify that circuit breaker triggered."""
        if self.burt and await self.burt.notify_circuit_breaker(daily_loss, limit):
            return

        notif = Notification(
            title="🚨 Circuit Breaker Triggered",
            description=f"Daily loss ${daily_loss:.2f} >= limit ${limit:.2f}",
            color=self.COLORS["circuit"],
        )
        await self._send_webhook(notif)

    async def notify_daily_summary(self, wins: int, losses: int,
                                    pnl: float) -> None:
        """Send end-of-day summary."""
        if self.burt and await self.burt.notify_daily_summary(wins, losses, pnl):
            return
        notif = Notification(
            title="📊 Daily Summary",
            description=f"{wins}W / {losses}L | P&L: ${pnl:+.2f}",
            color=self.COLORS["info"],
        )
        await self._send_webhook(notif)

    async def notify_alert(self, title: str, message: str,
                           color: str = "warning") -> None:
        """Generic operational alert (reconciliation divergence, missed stop...).

        These had no Burt branch and went straight to a webhook that is not
        configured, so every P0 alert — adopted position, missing exchange
        stop, externally-closed position — was silently discarded.
        """
        if self.burt and await self.burt.notify_alert(title, message):
            return
        notif = Notification(
            title=f"⚠️ {title}",
            description=message,
            color=self.COLORS.get(color, self.COLORS["warning"]),
        )
        await self._send_webhook(notif)

    # ------------------------------------------------------------------
    # Webhook sender
    # ------------------------------------------------------------------

    async def _send_webhook(self, notif: Notification) -> None:
        if not self._has_webhook:
            logger.warning(
                f"No delivery path for notification (Burt unavailable and "
                f"DISCORD_WEBHOOK_URL unset) — dropped: {notif.title}"
            )
            return
        embed = {
            "title": notif.title,
            "description": notif.description,
            "color": notif.color,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        }
        if notif.fields:
            embed["fields"] = notif.fields
        payload = {"embeds": [embed]}
        try:
            resp = await self._client.post(
                self.cfg.discord_webhook_url,
                json=payload,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning(f"Webhook send failed: {exc}")

    async def close(self) -> None:
        await self._client.aclose()
