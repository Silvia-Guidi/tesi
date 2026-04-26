"""
Quick check on the partial checkpoint, AWARE of pre-allocated storage.

Why this script exists
----------------------
main.py pre-allocates N_KEEP arrays at startup, e.g.
    samples['h'] = np.zeros((T, 3000), dtype=np.float32)
At every Gibbs iteration t >= BURNIN, column k = t - BURNIN is filled.
At a partial checkpoint, save_samples writes the WHOLE array to disk —
including the still-empty columns at the end. The previous version of
this script blindly took 'last 25%' of all columns and saw zeros, then
falsely reported a collapsing chain.

This version detects the number of actually-written columns by looking
for the last non-zero entry in phi_h_samples (a scalar that is exactly
0.0 only with probability 0), and trims the arrays to that length
before doing any statistics.

Run from the project root: python check_partial.py
"""

from __future__ import annotations
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("outputs")


def _load(name: str) -> np.ndarray:
    """Try the partial file first, then the final file."""
    for suffix in ("_partial", ""):
        path = OUTPUT_DIR / f"{name}_samples{suffix}.npy"
        if path.exists():
            return np.load(path)
    raise FileNotFoundError(f"Neither partial nor final {name} samples found")


def _detect_filled_length(phi_h_samples: np.ndarray) -> int:
    """
    Return the number of columns that have actually been written.
    A column k is unfilled if phi_h_samples[k] is exactly 0.0
    (storage was pre-allocated to zero, and the probability of a real
    posterior draw of phi_h landing exactly at 0.0 is zero).
    """
    nonzero = np.where(phi_h_samples != 0.0)[0]
    if nonzero.size == 0:
        return 0
    return int(nonzero[-1]) + 1


def main() -> None:
    h_full        = _load("h")            # (T, N_KEEP_alloc)
    phi_h_full    = _load("phi_h")        # (N_KEEP_alloc,)
    mu_h_full     = _load("mu_h")
    sigma_h2_full = _load("sigma_h2")
    lam_full      = _load("lambda_t")

    T, N_alloc = h_full.shape
    N_filled   = _detect_filled_length(phi_h_full)

    print(f"Pre-allocated draws : {N_alloc}")
    print(f"Filled draws so far : {N_filled}")
    if N_filled == 0:
        print("No draws yet. Wait for the first checkpoint after BURNIN.")
        return
    if N_filled < N_alloc:
        print(f"  -> trimming arrays to first {N_filled} columns")
    print()

    # Trim everything to the actually-filled portion
    h        = h_full[:, :N_filled]
    phi_h    = phi_h_full[:N_filled]
    mu_h     = mu_h_full[:N_filled]
    sigma_h2 = sigma_h2_full[:N_filled]
    lam      = lam_full[:, :N_filled]

    # ------------------------------------------------------------------
    # 1. std of h_t across time, per draw
    # ------------------------------------------------------------------
    std_per_draw = h.std(axis=0)
    print("=== std of h_t across time, per draw ===")
    print(f"  mean across draws : {std_per_draw.mean():.3f}")
    print(f"  min               : {std_per_draw.min():.3f}")
    print(f"  q05 / q50 / q95   : "
          f"{np.quantile(std_per_draw, 0.05):.3f} / "
          f"{np.quantile(std_per_draw, 0.50):.3f} / "
          f"{np.quantile(std_per_draw, 0.95):.3f}")

    # ------------------------------------------------------------------
    # 2. Trend across the chain: first 25% vs last 25% of FILLED draws
    # ------------------------------------------------------------------
    q = max(N_filled // 4, 1)
    print("\n=== std of h_t: first 25% vs last 25% of filled draws ===")
    print(f"  first 25%  mean std : {std_per_draw[:q].mean():.3f}")
    print(f"  last  25%  mean std : {std_per_draw[-q:].mean():.3f}")
    if std_per_draw[-q:].mean() < 0.5 * std_per_draw[:q].mean():
        print("  ** h_t is SHRINKING across iterations: this is the bug")
    else:
        print("  -> std is stable across the chain")

    # ------------------------------------------------------------------
    # 3. phi_h trajectory
    # ------------------------------------------------------------------
    print("\n=== phi_h trajectory ===")
    print(f"  mean              : {phi_h.mean():+.3f}")
    print(f"  first 25% mean    : {phi_h[:q].mean():+.3f}")
    print(f"  last  25% mean    : {phi_h[-q:].mean():+.3f}")
    print(f"  fraction below 0.5: {(phi_h < 0.5).mean():.1%}")

    # ------------------------------------------------------------------
    # 4. sigma_h2 trajectory
    # ------------------------------------------------------------------
    print("\n=== sigma_h2 trajectory ===")
    print(f"  mean              : {sigma_h2.mean():.4f}")
    print(f"  first 25% mean    : {sigma_h2[:q].mean():.4f}")
    print(f"  last  25% mean    : {sigma_h2[-q:].mean():.4f}")
    if sigma_h2[-q:].mean() < 0.5 * sigma_h2[:q].mean():
        print("  ** sigma_h2 is COLLAPSING: h_t innovations vanishing")
    else:
        print("  -> sigma_h2 is stable across the chain")

    # ------------------------------------------------------------------
    # 5. Lambda check (centring is per-iteration on the median, so the
    #    median of log lam should be ~0 for every draw)
    # ------------------------------------------------------------------
    print("\n=== lambda_t check ===")
    print(f"  mean over draws and time : {lam.mean():.3f}")
    print(f"  max anywhere             : {lam.max():.2f}")
    print(f"  min anywhere             : {lam.min():.4f}")
    log_lam_median_per_draw = np.median(np.log(lam), axis=0)
    print(f"  median(log lam) per draw - mean : "
          f"{log_lam_median_per_draw.mean():+.4f}  "
          f"(should be ~0 by construction with median centring)")

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print()
    healthy = (
        std_per_draw[-q:].mean() > 0.05 and
        sigma_h2[-q:].mean()    > 0.001 and
        np.isfinite(log_lam_median_per_draw.mean())
    )
    if healthy:
        print("VERDICT: chain looks healthy on the SV block.")
    else:
        print("VERDICT: SV block has issues, see flags above.")


if __name__ == "__main__":
    main()