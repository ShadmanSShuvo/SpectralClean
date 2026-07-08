"""
benchmark.py — Phase 2.4
DFT vs FFT timing benchmark with log-scale plot.

Tests increasing N from 2^4 to 2^14.
Plots execution time vs N on a log-log scale.
This is a standalone deliverable — run directly: python benchmark.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from transforms import dft, fft, zero_pad_to_power2


def time_function(func, x: np.ndarray, repeats: int = 3) -> float:
    """
    Time a transform function over `repeats` runs.
    Returns the minimum elapsed time in seconds (min over repeats
    reduces noise from OS scheduling jitter).
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(x)
        end = time.perf_counter()
        times.append(end - start)
    return min(times)


def run_benchmark(
    n_min_exp: int = 4,
    n_max_exp: int = 14,
    repeats: int = 3,
    dft_max_exp: int = 11,   # DFT is O(N²) — cap at 2^11 to avoid minutes-long wait
) -> dict:
    """
    Benchmark DFT vs FFT across N = 2^n_min … 2^n_max.

    DFT is only timed up to 2^dft_max_exp (beyond that it would take
    too long and the O(N²) vs O(N log N) contrast is already clear).

    Returns
    -------
    dict with keys:
        Ns        — list of sizes tested
        fft_times — FFT times (seconds)
        dft_times — DFT times (seconds), None for N > dft_max
        theory_N2 — theoretical N² curve (normalised to first FFT point)
        theory_NlogN — theoretical N log N curve
    """
    exps = list(range(n_min_exp, n_max_exp + 1))
    Ns = [2 ** e for e in exps]

    fft_times = []
    dft_times = []

    print(f"\n{'N':>8}  {'FFT (ms)':>12}  {'DFT (ms)':>12}  {'Speedup':>10}")
    print("─" * 50)

    for e, N in zip(exps, Ns):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(N).astype(np.complex128)

        t_fft = time_function(fft, x, repeats) * 1000  # ms
        fft_times.append(t_fft)

        if e <= dft_max_exp:
            t_dft = time_function(dft, x, repeats) * 1000
            dft_times.append(t_dft)
            speedup = f"{t_dft / t_fft:.1f}×"
        else:
            dft_times.append(None)
            speedup = "(DFT too slow)"

        dft_str = f"{t_dft:.2f}" if e <= dft_max_exp else "—"
        print(f"{N:>8}  {t_fft:>10.2f}ms  {dft_str:>10}ms  {speedup:>10}")

    # Theoretical reference curves (anchored to first measured FFT point)
    Ns_arr = np.array(Ns, dtype=float)
    anchor = fft_times[0] / (Ns_arr[0] * np.log2(Ns_arr[0]))
    theory_NlogN = anchor * Ns_arr * np.log2(Ns_arr)

    anchor2 = (dft_times[0] if dft_times[0] is not None else fft_times[0]) / (Ns_arr[0] ** 2)
    theory_N2 = anchor2 * Ns_arr ** 2

    return {
        "Ns":           Ns,
        "fft_times":    fft_times,
        "dft_times":    dft_times,
        "theory_NlogN": theory_NlogN.tolist(),
        "theory_N2":    theory_N2.tolist(),
        "exps":         exps,
        "dft_max_exp":  dft_max_exp,
    }


def plot_benchmark(results: dict, save_path: str = None) -> plt.Figure:
    """
    Render the DFT vs FFT benchmark plot.

    - Log-log axes (both x and y)
    - FFT measured times (solid line)
    - DFT measured times (solid line, where available)
    - Theoretical O(N log N) and O(N²) dashed reference lines
    """
    Ns           = results["Ns"]
    fft_times    = results["fft_times"]
    dft_times    = results["dft_times"]
    theory_NlogN = results["theory_NlogN"]
    theory_N2    = results["theory_N2"]
    dft_max_exp  = results["dft_max_exp"]

    # Separate measured DFT points
    dft_Ns    = [n for n, t in zip(Ns, dft_times) if t is not None]
    dft_vals  = [t for t in dft_times if t is not None]

    # ── Figure layout ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    palette = {
        "fft":    "#00E5FF",
        "dft":    "#FF6B6B",
        "nlogn":  "#69FF47",
        "n2":     "#FFA726",
        "grid":   "#21262D",
        "text":   "#C9D1D9",
    }

    # Measured curves
    ax.loglog(Ns, fft_times, "o-",
              color=palette["fft"], linewidth=2.2, markersize=7,
              label="FFT (Cooley-Tukey, scratch)", zorder=5)

    if dft_vals:
        ax.loglog(dft_Ns, dft_vals, "s-",
                  color=palette["dft"], linewidth=2.2, markersize=7,
                  label=f"DFT (naive, scratch, N ≤ 2^{dft_max_exp})", zorder=5)

    # Theoretical reference lines
    ax.loglog(Ns, theory_NlogN, "--",
              color=palette["nlogn"], linewidth=1.4, alpha=0.75,
              label="O(N log N) reference", zorder=3)

    ax.loglog(Ns[:len(dft_vals) + 2], theory_N2[:len(dft_vals) + 2], "--",
              color=palette["n2"], linewidth=1.4, alpha=0.75,
              label="O(N²) reference", zorder=3)

    # Annotation: speedup at largest shared N
    if dft_vals and len(dft_vals) > 0:
        idx = len(dft_vals) - 1
        speedup = dft_vals[idx] / fft_times[idx]
        ax.annotate(
            f"~{speedup:.0f}× faster at N={dft_Ns[idx]:,}",
            xy=(dft_Ns[idx], dft_vals[idx]),
            xytext=(dft_Ns[idx] // 2, dft_vals[idx] * 4),
            fontsize=9, color=palette["text"],
            arrowprops=dict(arrowstyle="->", color=palette["text"], lw=1.2),
        )

    # ── Styling ────────────────────────────────────────────────────────
    ax.set_xlabel("Signal length N (samples)", color=palette["text"], fontsize=12)
    ax.set_ylabel("Execution time (ms)", color=palette["text"], fontsize=12)
    ax.set_title(
        "Benchmark: Naive DFT  vs  Cooley-Tukey FFT\n"
        "(both implemented from scratch — no library FFT used in pipeline)",
        color="white", fontsize=13, pad=12
    )

    ax.tick_params(colors=palette["text"], which="both")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    ax.grid(True, which="both", color=palette["grid"], linestyle="--", alpha=0.5)

    legend = ax.legend(
        facecolor="#1C2128", edgecolor="#30363D",
        labelcolor=palette["text"], fontsize=10
    )

    # Complexity annotation box
    note = (
        "DFT: O(N²) — 2 nested loops\n"
        "FFT: O(N log N) — recursive halving\n"
        "At N=2¹⁴: ~14× theoretical advantage"
    )
    ax.text(
        0.02, 0.97, note,
        transform=ax.transAxes,
        fontsize=8.5, color=palette["text"],
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1C2128",
                  edgecolor="#30363D", alpha=0.9),
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\n✓ Benchmark plot saved → {save_path}")

    return fig


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("SpectralClean — DFT vs FFT Benchmark")
    print("=" * 44)

    results = run_benchmark(n_min_exp=4, n_max_exp=14, dft_max_exp=11)
    fig = plot_benchmark(results, save_path="report/benchmark_dft_vs_fft.png")
    plt.show()
