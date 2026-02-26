"""Tests for implied volatility computation."""

import numpy as np
import pytest

from smart_index.features.implied_vol import bs_price, bs_vega, implied_vol


class TestBSPrice:
    """Verify Black-Scholes pricing against known values."""

    def test_atm_call(self):
        # ATM call: S=K=100, T=1, r=0, sigma=0.20
        price = bs_price(100, 100, 1.0, 0.0, 0.20, "C")
        # Expected ~7.97 (standard BS result)
        assert 7.5 < price < 8.5

    def test_put_call_parity(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25
        call = bs_price(S, K, T, r, sigma, "C")
        put = bs_price(S, K, T, r, sigma, "P")
        # C - P = S - K*exp(-rT)
        parity = S - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 1e-10

    def test_deep_itm_call(self):
        # Deep ITM call should be close to intrinsic
        price = bs_price(150, 100, 0.01, 0.0, 0.20, "C")
        assert abs(price - 50.0) < 0.5

    def test_zero_time(self):
        assert bs_price(110, 100, 0, 0, 0.2, "C") == 10.0
        assert bs_price(90, 100, 0, 0, 0.2, "P") == 10.0


class TestBSVega:
    def test_atm_vega_positive(self):
        v = bs_vega(100, 100, 1.0, 0.0, 0.20)
        assert v > 0

    def test_vega_zero_at_expiry(self):
        assert bs_vega(100, 100, 0, 0, 0.20) == 0.0


class TestImpliedVol:
    def test_roundtrip_brentq(self):
        """Price → IV → Price should roundtrip."""
        true_vol = 0.25
        S, K, T, r = 100, 105, 0.5, 0.02
        price = bs_price(S, K, T, r, true_vol, "C")
        recovered = implied_vol(price, S, K, T, r, "C", method="brentq")
        assert recovered is not None
        assert abs(recovered - true_vol) < 1e-6

    def test_roundtrip_newton(self):
        true_vol = 0.30
        S, K, T, r = 100, 95, 0.25, 0.01
        price = bs_price(S, K, T, r, true_vol, "P")
        recovered = implied_vol(price, S, K, T, r, "P", method="newton")
        assert recovered is not None
        assert abs(recovered - true_vol) < 1e-5

    def test_below_intrinsic_returns_none(self):
        # Price below intrinsic should fail gracefully
        result = implied_vol(0.5, 110, 100, 0.5, 0.0, "C")
        assert result is None

    def test_zero_dte_returns_none(self):
        result = implied_vol(5.0, 100, 100, 0.0, 0.0, "C")
        assert result is None

    @pytest.mark.parametrize("vol", [0.05, 0.15, 0.30, 0.60, 1.00])
    def test_range_of_vols(self, vol):
        """Solver should handle low to very high vols."""
        price = bs_price(100, 100, 0.5, 0.0, vol, "C")
        recovered = implied_vol(price, 100, 100, 0.5, 0.0, "C")
        assert recovered is not None
        assert abs(recovered - vol) < 1e-5
