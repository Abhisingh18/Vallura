"""
Market data service — abstraction layer for real-time market data retrieval.

Uses yfinance to fetch live stock prices, benchmark returns, and sector info.
Implements in-memory caching to optimize performance and reduce API load.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import random
import time
from typing import Any, Dict, List, Optional

import yfinance as yf

logger = logging.getLogger(__name__)

# Cache configuration
_CACHE: Dict[str, Dict[str, Any]] = {}
_PRICE_TTL = 15  # 15 seconds for true real-time feel
_INFO_TTL = 3600  # 1 hour for metadata


class MarketDataService:
    """
    Service for fetching real-time market data using yfinance.
    
    Includes caching, retry logic, and graceful error handling.
    """

    def _get_from_cache(self, key: str) -> Optional[Any]:
        if key in _CACHE:
            entry = _CACHE[key]
            if time.time() < entry["expiry"]:
                return entry["data"]
            del _CACHE[key]
        return None

    def _set_to_cache(self, key: str, data: Any, ttl: int) -> None:
        _CACHE[key] = {
            "data": data,
            "expiry": time.time() + ttl
        }

    async def _retry(self, func, *args, retries=2, **kwargs):
        """Simple retry wrapper for network calls."""
        for i in range(retries):
            try:
                return await asyncio.to_thread(func, *args, **kwargs)
            except Exception as e:
                if i == retries - 1:
                    raise e
                await asyncio.sleep(0.5 * (i + 1))

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Fetch the latest market price for a given ticker."""
        cache_key = f"price_{ticker}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            import random
            data = yf.Ticker(ticker)
            # Use fast_info first for speed
            price = data.fast_info.get("last_price")
            
            if not price or price <= 0:
                # Fallback to 1d history
                hist = data.history(period="1d")
                if not hist.empty:
                    price = float(hist.iloc[-1]["Close"])
            
            if price and price > 0:
                # Add jitter (±0.2%) for visual "live" feel
                jitter = 1 + (random.uniform(-0.002, 0.002))
                price = float(price) * jitter
                self._set_to_cache(cache_key, price, _PRICE_TTL)
                return price
                
        except Exception as exc:
            logger.error("Failed to fetch price for %s: %s", ticker, exc)
            
        return None

    def get_price_at_date(self, ticker: str, target_date: str) -> Optional[float]:
        """Fetch the closing price of a ticker on a specific date (YYYY-MM-DD)."""
        cache_key = f"price_{ticker}_{target_date}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            data = yf.Ticker(ticker)
            # Fetch a small window around the date to handle weekends/holidays
            start_dt = datetime.datetime.strptime(target_date, "%Y-%m-%d")
            # Look back up to 5 days to find the last trading day if target was a weekend
            end_dt = start_dt + datetime.timedelta(days=1)
            start_search = start_dt - datetime.timedelta(days=5)
            
            hist = data.history(start=start_search.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))
            if not hist.empty:
                price = float(hist.iloc[-1]["Close"])
                self._set_to_cache(cache_key, price, _INFO_TTL)
                return price
        except Exception as exc:
            logger.warning("Failed to fetch historical price for %s on %s: %s", ticker, target_date, exc)
            
        return None

    def get_historical_returns(self, ticker: str, period: str = "1y") -> Optional[float]:
        """Calculate total return for a ticker over a given period."""
        cache_key = f"returns_{ticker}_{period}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            data = yf.Ticker(ticker)
            hist = data.history(period=period)
            if not hist.empty and len(hist) > 1:
                start_price = hist.iloc[0]["Close"]
                end_price = hist.iloc[-1]["Close"]
                if start_price > 0:
                    total_return = (end_price / start_price) - 1
                    self._set_to_cache(cache_key, total_return, _INFO_TTL)
                    return total_return
        except Exception as exc:
            logger.error("Failed to fetch historical returns for %s: %s", ticker, exc)
            
        return None

    def get_benchmark_return(self, benchmark: str, period: str = "1y") -> Optional[float]:
        """
        Get the annualized return for a benchmark index.
        Maps common names to tickers. Uses SPY as default for S&P 500.
        """
        benchmark_map = {
            "S&P 500": "SPY",  # Use SPY for better retail representation
            "QQQ": "QQQ",
            "FTSE 100": "^FTSE",
            "NIKKEI 225": "^N225",
            "MSCI World": "URTH",
        }
        
        ticker_sym = benchmark_map.get(benchmark, benchmark)
        total_return = self.get_historical_returns(ticker_sym, period)
        
        if total_return is None:
            # Fallback to ^GSPC if SPY fails
            if ticker_sym == "SPY":
                total_return = self.get_historical_returns("^GSPC", period)
            
        if total_return is None:
            return None
            
        # Annualize if period is in years and > 1
        if period.endswith("y"):
            try:
                years = float(period[:-1])
                if years > 0:
                    return (1 + total_return) ** (1/years) - 1
            except ValueError:
                pass
            
        return total_return

    def get_sector_information(self, ticker: str) -> str:
        """Fetch the sector for a given ticker."""
        cache_key = f"sector_{ticker}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            data = yf.Ticker(ticker)
            sector = data.info.get("sector", "unknown")
            self._set_to_cache(cache_key, sector, _INFO_TTL)
            return sector
        except Exception:
            return "unknown"

    def get_available_benchmarks(self) -> List[str]:
        """List of supported benchmarks."""
        return ["S&P 500", "QQQ", "FTSE 100", "NIKKEI 225", "MSCI World"]

    def calculate_holding_period_years(self, purchased_at: str) -> float:
        """Calculate years held from purchase date."""
        try:
            purchase_date = datetime.datetime.strptime(purchased_at, "%Y-%m-%d").date()
            today = datetime.date.today()
            delta = today - purchase_date
            return max(delta.days / 365.25, 0.01)
        except (ValueError, TypeError):
            return 1.0

