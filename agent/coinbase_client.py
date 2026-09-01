"""
Coinbase Brokerage v3 Native Client
====================================

Thin client that hits /api/v3/brokerage/ directly with JWT (ES256) auth.
No CCXT — FCM perp products are not exposed by ccxt.coinbase.

Auth flow (per request, JWT valid ~120s):
  1. Build payload: sub, iss='cdp', nbf, exp, uri
  2. Sign with EC private key (PEM from CDP keypair) using ES256
  3. Set Authorization: Bearer {jwt}
"""

import asyncio
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
import jwt
from cryptography.hazmat.primitives import serialization
from loguru import logger

import config

API_BASE = "https://api.coinbase.com"


@dataclass(frozen=True)
class CbFutureProduct:
    product_id: str
    display_name: str
    status: str
    trading_enabled: bool
    price: float | None
    volume_24h: float | None
    max_leverage: float | None
    funding_rate: float | None
    open_interest: float | None
    mark_price: float | None


@dataclass(frozen=True)
class CbCandle:
    start: int   # unix epoch seconds
    low: float
    high: float
    open: float
    close: float
    volume: float


class CoinbaseClient:
    """Native Coinbase Brokerage v3 client with JWT auth."""

    PUBLIC_RATE = 15
    PRIVATE_RATE = 10

    def __init__(self):
        self.cfg = config.get_config()
        self._session: aiohttp.ClientSession | None = None
        self._key_name = self.cfg.coinbase_api_key
        self._private_key_pem = self.cfg.coinbase_api_secret.replace("\\n", "\n")
        self._private_key = self._load_private_key()
        self._sem_public = asyncio.Semaphore(self.PUBLIC_RATE)
        self._sem_private = asyncio.Semaphore(self.PRIVATE_RATE)
    
    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _load_private_key(self):
        """Parse PEM EC private key."""
        try:
            return serialization.load_pem_private_key(
                self._private_key_pem.encode(), password=None
            )
        except Exception:
            logger.error("Failed to parse COINBASE_API_SECRET — is it valid PEM?")
            return None

    def _build_jwt(self, method: str, path: str) -> str:
        """Build a 120s JWT for a single request."""
        now = int(time.time())
        payload = {
            "sub": self._key_name,
            "iss": "cdp",
            "nbf": now,
            "exp": now + 120,
            "uri": f"{method} api.coinbase.com{path}",
        }
        return jwt.encode(
            payload,
            self._private_key,
            algorithm="ES256",
            headers={"kid": self._key_name, "nonce": secrets.token_hex(16)},
        )

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        """Make authenticated request with rate-limit semaphore + auto-retry."""
        await self._ensure_session()
        url = f"{API_BASE}{path}"
        sem = self._sem_private if self._key_name else self._sem_public

        async with sem:
            headers = kwargs.pop("headers", {})
            if self._private_key:
                token = self._build_jwt(method, path)
                headers["Authorization"] = f"Bearer {token}"

            for attempt in range(3):
                try:
                    async with self._session.request(
                        method, url, headers=headers, **kwargs
                    ) as resp:
                        text = await resp.text()
                        if resp.status == 429:
                            wait = 2 ** attempt
                            logger.warning(f"Rate limited — retry in {wait}s")
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        if not text:
                            return {}
                        return json.loads(text)
                except aiohttp.ClientError as exc:
                    if attempt < 2:
                        await asyncio.sleep(1)
                        continue
                    raise

    # ------------------------------------------------------------------
    # Product discovery
    # ------------------------------------------------------------------

    async def list_future_products(self) -> list[CbFutureProduct]:
        """List all FUTURE products, filter to perp-style."""
        data = await self._request(
            "GET", "/api/v3/brokerage/market/products",
            params={"product_type": "FUTURE", "limit": 300}
        )
        products = data.get("products", [])
        perps = []
        for p in products:
            dn = p.get("display_name", "")
            if "PERP" in dn:
                # Leverage = 1 / margin_rate from intraday_margin_rate
                margin = p.get("future_product_details", {}).get("intraday_margin_rate", {})
                long_margin = margin.get("long_margin_rate", "")
                max_lev = 1.0 / float(long_margin) if long_margin else 10.0
                
                perps.append(
                    CbFutureProduct(
                        product_id=p["product_id"],
                        display_name=dn,
                        status=p.get("status", ""),
                        trading_enabled=not p.get("trading_disabled", True),
                        price=float(p["price"]) if p.get("price") else None,
                        # Coinbase sends "" for empty numerics — float("") raises.
                        volume_24h=float(p["approximate_quote_24h_volume"])
                            if p.get("approximate_quote_24h_volume") else None,
                        max_leverage=max_lev,
                        funding_rate=None,
                        open_interest=None,
                        mark_price=float(p["price"]) if p.get("price") else None,
                    )
                )
        logger.info(f"Discovered {len(perps)} perp products on FCM")
        return perps

    async def hydrate_product_details(self, product_id: str) -> dict:
        """Fetch per-product detail for funding, OI, contract size, etc."""
        data = await self._request(
            "GET",
            f"/api/v3/brokerage/market/products/{product_id}",
        )
        fp = data.get("future_product_details", {})
        pdets = fp.get("perpetual_details", {})
        funding = fp.get("funding_rate", "") or pdets.get("funding_rate", "")
        oi = fp.get("open_interest", "") or pdets.get("open_interest", "")
        contract_size = fp.get("contract_size", "")
        # `price` is the product's current mark. PositionMonitor reads
        # "mark_price" from this dict on every tick — without it, every
        # position marks at its entry price and NO stop, take-profit, or
        # trailing ratchet can ever fire.
        mark = data.get("price", "")
        return {
            "funding_rate": float(funding) if funding else None,
            "open_interest": float(oi) if oi else None,
            "max_leverage": None,
            "contract_size": float(contract_size) if contract_size else None,
            "mark_price": float(mark) if mark else None,
        }

    async def hydrate_all(self, products: list[CbFutureProduct]) -> list[CbFutureProduct]:
        """Hydrate funding/OI for a list of products (concurrent)."""
        sem = asyncio.Semaphore(4)

        async def _hydrate(p: CbFutureProduct) -> CbFutureProduct:
            async with sem:
                try:
                    d = await self.hydrate_product_details(p.product_id)
                    return CbFutureProduct(
                        product_id=p.product_id,
                        display_name=p.display_name,
                        status=p.status,
                        trading_enabled=p.trading_enabled,
                        price=p.price,
                        volume_24h=p.volume_24h,
                        max_leverage=d.get("max_leverage") or p.max_leverage,
                        funding_rate=d.get("funding_rate"),
                        open_interest=d.get("open_interest"),
                        mark_price=p.mark_price,
                    )
                except Exception as exc:
                    logger.warning(f"Failed to hydrate {p.product_id}: {exc}")
                    return p

        return await asyncio.gather(*(_hydrate(p) for p in products))

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------

    async def get_candles(
        self,
        product_id: str,
        granularity: str = "FIFTEEN_MINUTE",
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> list[CbCandle]:
        """Fetch OHLCV candles. Max 300 per request."""
        params = {"granularity": granularity}
        if start:
            params["start"] = str(start)
        if end:
            params["end"] = str(end)
        if limit:
            params["limit"] = str(limit)

        data = await self._request(
            "GET",
            f"/api/v3/brokerage/market/products/{product_id}/candles",
            params=params,
        )
        raw = data.get("candles", [])
        # Response shape: {"start": str, "low": str, "high": str, "open": str, "close": str, "volume": str}
        candles = []
        for r in raw:
            candles.append(
                CbCandle(
                    start=int(r["start"]),
                    low=float(r["low"]),
                    high=float(r["high"]),
                    open=float(r["open"]),
                    close=float(r["close"]),
                    volume=float(r["volume"]),
                )
            )
        # Sort ascending by time
        candles.sort(key=lambda c: c.start)
        return candles

    async def get_candles_multi(
        self,
        product_ids: list[str],
        granularity: str = "ONE_HOUR",
        max_concurrent: int = 8,
    ) -> dict[str, list[CbCandle]]:
        """Fetch candles for multiple products concurrently."""
        sem = asyncio.Semaphore(max_concurrent)
        results: dict[str, list[CbCandle]] = {}

        async def _fetch(pid: str) -> None:
            async with sem:
                try:
                    results[pid] = await self.get_candles(pid, granularity)
                except Exception as exc:
                    logger.warning(f"Failed candles for {pid}: {exc}")
                    results[pid] = []

        await asyncio.gather(*(_fetch(pid) for pid in product_ids))
        return results

    # ------------------------------------------------------------------
    # Account / balance / positions
    # ------------------------------------------------------------------

    async def get_accounts(self) -> list[dict]:
        """Get spot account balances."""
        data = await self._request("GET", "/api/v3/brokerage/accounts")
        return data.get("accounts", [])

    async def get_futures_balance_summary(self) -> dict:
        """Get CFM balance summary — source of truth for trade capacity."""
        data = await self._request("GET", "/api/v3/brokerage/cfm/balance_summary")
        return data or {}

    async def get_futures_positions(self) -> list[dict]:
        """List open CFM positions."""
        data = await self._request("GET", "/api/v3/brokerage/cfm/positions")
        return data.get("positions", [])

    # ------------------------------------------------------------------
    # Orders (CFM futures)
    # ------------------------------------------------------------------

    async def place_futures_market_order(
        self, product_id: str, side: str, contracts: int,
        leverage: int = 1, margin_type: str = "ISOLATED",
    ) -> dict:
        """
        Market entry for a CFM future. `contracts` is whole contracts
        (base_size is contract count for futures, not base currency).
        """
        payload = {
            "client_order_id": secrets.token_hex(16),
            "product_id": product_id,
            "side": side,
            "order_configuration": {
                "market_market_ioc": {"base_size": str(contracts)},
            },
            "leverage": str(leverage),
            "margin_type": margin_type,
        }
        return await self.place_order(payload)

    async def place_futures_stop_order(
        self, product_id: str, side: str, contracts: int,
        stop_price: float, limit_price: float, stop_direction: str,
    ) -> dict:
        """
        Exchange-native stop-limit (crash protection for live positions).
        `stop_direction` is STOP_DIRECTION_STOP_DOWN (long SL) or
        STOP_DIRECTION_STOP_UP (short SL).
        """
        payload = {
            "client_order_id": secrets.token_hex(16),
            "product_id": product_id,
            "side": side,
            "order_configuration": {
                "stop_limit_stop_limit_gtc": {
                    "base_size": str(contracts),
                    "limit_price": f"{limit_price:.8f}".rstrip("0").rstrip("."),
                    "stop_price": f"{stop_price:.8f}".rstrip("0").rstrip("."),
                    "stop_direction": stop_direction,
                },
            },
        }
        return await self.place_order(payload)

    async def close_futures_position(self, product_id: str, size: str | None = None) -> dict:
        """
        Close a CFM position via the dedicated close-position endpoint
        (marketable, reduce-only). `size` = contracts for partial close;
        omit for full close.
        """
        payload: dict = {
            "client_order_id": secrets.token_hex(16),
            "product_id": product_id,
        }
        if size is not None:
            payload["size"] = size
        return await self._request(
            "POST", "/api/v3/brokerage/orders/close_position", json=payload
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_order(self, payload: dict) -> dict:
        """Place an order on the exchange."""
        return await self._request(
            "POST", "/api/v3/brokerage/orders", json=payload
        )

    async def cancel_orders(self, order_ids: list[str]) -> dict:
        """Cancel orders by IDs."""
        return await self._request(
            "POST",
            "/api/v3/brokerage/orders/batch_cancel",
            json={"order_ids": order_ids},
        )

    async def list_open_orders(self) -> list[dict]:
        """List currently open orders."""
        data = await self._request(
            "GET",
            "/api/v3/brokerage/orders/historical/batch",
            params={"order_status": "OPEN"},
        )
        return data.get("orders", [])

    async def get_fills(self) -> list[dict]:
        """Get recent fills."""
        data = await self._request(
            "GET", "/api/v3/brokerage/orders/historical/fills"
        )
        return data.get("fills", [])

    # ------------------------------------------------------------------
    # Sweeps (spot ↔ futures)
    # ------------------------------------------------------------------

    async def schedule_sweep(self, amount: str, currency: str = "USD") -> dict:
        """Schedule a sweep from spot to futures."""
        return await self._request(
            "POST",
            "/api/v3/brokerage/cfm/sweeps/schedule",
            json={"amount": amount, "currency": currency},
        )

    async def get_sweeps(self) -> list[dict]:
        """Get pending sweeps."""
        data = await self._request("GET", "/api/v3/brokerage/cfm/sweeps")
        return data.get("sweeps", [])

    # ------------------------------------------------------------------
    # Connectivity checks
    # ------------------------------------------------------------------

    async def verify_auth(self) -> bool:
        """Quick auth check via /accounts."""
        try:
            await self.get_accounts()
            logger.info("Coinbase API auth verified")
            return True
        except Exception as exc:
            logger.error(f"Coinbase auth failed: {exc}")
            return False

    async def verify_futures_provisioned(self) -> bool:
        """Check if CFM account exists and has buying power."""
        try:
            summary = await self.get_futures_balance_summary()
            bp = float(summary.get("futures_buying_power", {}).get("value", 0))
            logger.info(f"Futures buying power: ${bp:.2f}")
            return True
        except Exception as exc:
            logger.error(f"CFM not provisioned or funded: {exc}")
            return False
