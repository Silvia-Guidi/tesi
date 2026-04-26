"""
Quick check of how heavy-tailed the data residuals are.

If the ratio q99(res^2) / median(res^2) is very large (>50), it means
the AR(1) residuals have spikes that lambda_t (the Student-t mixing
weight) will absorb. In that regime, h_t (the persistent stochastic
volatility) gets little signal and phi_h becomes weakly identified.

A phi_h around 0.5-0.7 is then NOT a sampler bug — it is the posterior
that the data actually support. Macroeconomic quarterly data tend to
support phi_h ~ 0.95; daily energy-price data tend to support phi_h
much lower because most of the variance is local (jumps, market events).

Run from the project root: python check_data_tails.py
"""

from __future__ import annotations
import numpy as np
from pathlib import Path


def main() -> None:
    Y = np.load(Path("data") / "Y.npy")
    print(f"Y shape: {Y.shape}")
    print()

    T, ny = Y.shape

    # Univariate AR(1) per series, OLS in closed form.
    print(f"{'series':>6} | {'AR1 coef':>9} | {'res std':>8} | "
          f"{'q99/med':>8} | {'comment':<30}")
    print("-" * 78)

    ratios = []
    for j in range(ny):
        y  = Y[1:, j]
        yl = Y[:-1, j]
        X  = np.column_stack([np.ones(len(yl)), yl])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b
        sq = res * res

        ar1 = float(b[1])
        sd  = float(np.sqrt(sq.mean()))
        med = float(np.median(sq))
        q99 = float(np.quantile(sq, 0.99))
        ratio = q99 / med if med > 0 else float("inf")
        ratios.append(ratio)

        if ratio > 100:
            comment = "VERY heavy tails"
        elif ratio > 50:
            comment = "heavy tails"
        elif ratio > 20:
            comment = "moderate tails"
        else:
            comment = "ok"

        print(f"{j:>6} | {ar1:>+9.3f} | {sd:>8.3f} | {ratio:>8.1f} | {comment}")

    print()
    print(f"Median ratio across series: {np.median(ratios):.1f}")
    print(f"Max ratio across series:    {max(ratios):.1f}")
    print()

    if np.median(ratios) > 50:
        print("=" * 78)
        print("DIAGNOSIS: your data have very heavy-tailed residuals.")
        print("In this regime, lambda_t absorbs most of the variance and h_t")
        print("has little signal left. A posterior phi_h ~ 0.5-0.7 is then")
        print("the truth, NOT a sampler bug.")
        print()
        print("Options:")
        print("  1. Accept the data-driven phi_h and move on.")
        print("  2. Use a less informative prior on phi_h, e.g. Beta(2, 2)")
        print("     (mean 0.5, broad) instead of Beta(20, 1.5) (mean 0.93).")
        print("  3. Pre-clean obvious spikes in the data before fitting.")
        print("=" * 78)
    elif np.median(ratios) > 20:
        print("Moderate tails — phi_h around 0.7-0.9 is plausible.")
    else:
        print("Tails look fine — phi_h should be high if the model is correct.")


if __name__ == "__main__":
    main()