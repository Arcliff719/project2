"""
Analysis and evaluation tools for lossy source coding.
Evaluates: PSNR, compression ratio, algorithm complexity, rate-distortion.
"""

import numpy as np
import time
from typing import List, Dict
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from .source_coding import encode_image, decode_image, measure_complexity


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (in dB).
    Higher = better quality. Typical range: 20–50 dB for lossy image coding.
    """
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)


def evaluate_quality_range(img: np.ndarray,
                           qualities: List[int] = None) -> List[Dict]:
    """
    Evaluate source coding across a range of quality factors.

    Args:
        img: 2D grayscale uint8 image
        qualities: list of quality factors (1–100). Default: [1, 5, 10, 25, 50, 75, 90, 100]

    Returns:
        List of dicts with metrics per quality level.
    """
    if qualities is None:
        qualities = [1, 5, 10, 25, 50, 75, 90, 100]

    results = []
    for q in qualities:
        encoded = encode_image(img, q)
        decoded = decode_image(encoded)
        p = psnr(img, decoded)

        # Complexity
        comp = measure_complexity(img, q)

        results.append({
            "quality": q,
            "psnr_db": round(p, 3),
            "mse": round(mse(img, decoded), 4),
            "compression_ratio": round(encoded.compression_ratio, 3),
            "bpp": round(encoded.bpp, 3),
            "original_bits": encoded.original_size_bits,
            "compressed_bits": encoded.compressed_size_bits,
            "encode_time_s": round(comp["encode_time_s"], 4),
            "decode_time_s": round(comp["decode_time_s"], 4),
            "encode_memory_kb": round(comp["encode_memory_peak_kb"], 2),
            "decode_memory_kb": round(comp["decode_memory_peak_kb"], 2),
        })
        print(f"  Q={q:3d}: PSNR={p:6.2f} dB, "
              f"Ratio={encoded.compression_ratio:6.2f}x, "
              f"BPP={encoded.bpp:5.3f}, "
              f"Enc={comp['encode_time_s']:.4f}s, "
              f"Dec={comp['decode_time_s']:.4f}s")

    return results


def plot_rate_distortion(results: List[Dict], save_path: str = "rate_distortion.png"):
    """Plot Rate-Distortion curve (BPP vs PSNR)."""
    bpps = [r["bpp"] for r in results]
    psnrs = [r["psnr_db"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bpps, psnrs, "bo-", linewidth=2, markersize=8)
    for i, r in enumerate(results):
        ax.annotate(f"Q={r['quality']}", (bpps[i], psnrs[i]),
                    textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set_xlabel("Bits per Pixel (BPP)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Rate-Distortion Curve", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] Rate-distortion curve -> {save_path}")


def plot_complexity(results: List[Dict], save_path: str = "complexity.png"):
    """Plot encoding/decoding time vs quality."""
    qualities = [r["quality"] for r in results]
    enc_times = [r["encode_time_s"] for r in results]
    dec_times = [r["decode_time_s"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(np.arange(len(qualities)) - 0.15, enc_times, 0.3, label="Encode", color="#2196F3")
    ax1.bar(np.arange(len(qualities)) + 0.15, dec_times, 0.3, label="Decode", color="#4CAF50")
    ax1.set_xticks(range(len(qualities)))
    ax1.set_xticklabels([f"Q={q}" for q in qualities])
    ax1.set_ylabel("Time (seconds)", fontsize=12)
    ax1.set_title("Encode / Decode Time", fontsize=13)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(qualities, enc_times, "o-", label="Encode", color="#2196F3")
    ax2.plot(qualities, dec_times, "s-", label="Decode", color="#4CAF50")
    ax2.set_xlabel("Quality Factor", fontsize=12)
    ax2.set_ylabel("Time (seconds)", fontsize=12)
    ax2.set_title("Time vs Quality Factor", fontsize=13)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] Complexity plots -> {save_path}")


def plot_compression_summary(results: List[Dict], save_path: str = "compression_summary.png"):
    """Plot compression ratio and PSNR vs quality in one figure."""
    qualities = [r["quality"] for r in results]
    ratios = [r["compression_ratio"] for r in results]
    psnrs = [r["psnr_db"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1, color2 = "#E91E63", "#2196F3"

    ax1.bar(qualities, ratios, width=3, color=color1, alpha=0.7, label="Compression Ratio")
    ax1.set_xlabel("Quality Factor", fontsize=12)
    ax1.set_ylabel("Compression Ratio", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(qualities, psnrs, "s-", color=color2, linewidth=2, markersize=8, label="PSNR")
    ax2.set_ylabel("PSNR (dB)", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Compression Performance vs Quality Factor", fontsize=14)
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] Compression summary -> {save_path}")


def print_analysis_table(results: List[Dict]):
    """Print a formatted analysis table."""
    header = (f"{'Q':>4s}  {'PSNR(dB)':>9s}  {'Ratio':>8s}  {'BPP':>7s}  "
              f"{'Enc(ms)':>8s}  {'Dec(ms)':>8s}  {'EncMem(KB)':>11s}  {'DecMem(KB)':>11s}")
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        print(f"{r['quality']:4d}  {r['psnr_db']:9.2f}  {r['compression_ratio']:8.2f}  "
              f"{r['bpp']:7.4f}  {r['encode_time_s']*1000:8.2f}  {r['decode_time_s']*1000:8.2f}  "
              f"{r['encode_memory_kb']:11.2f}  {r['decode_memory_kb']:11.2f}")
    print(sep)


def complexity_analysis(img: np.ndarray) -> Dict:
    """
    Detailed algorithmic complexity analysis.

    For N×N image divided into (N/8)² blocks:
    - DCT/IDCT: O(K · 8²) per block where K=8² for naive 2D DCT
      → O(N² · 64) = O(N²) overall
    - Quantization: O(1) per coefficient → O(N²)
    - Huffman encode: O(M log M) where M = number of unique symbols
    - Huffman decode: O(L) where L = bitstream length

    Space complexity: O(N²) for image storage + O(S) for symbol tables.
    """
    h, w = img.shape
    n_blocks = ((h + 7) // 8) * ((w + 7) // 8)

    analysis = {
        "image_dimensions": f"{h}x{w}",
        "total_pixels": h * w,
        "n_8x8_blocks": n_blocks,
        "dct_ops_per_block": 8**4,  # 8x8 x 8x8 = 4096 (naive)
        "total_dct_ops": n_blocks * (8**4),
        "time_complexity_encode": "O(N^2) - dominated by DCT on each 8x8 block",
        "time_complexity_decode": "O(N^2) - symmetric to encoder",
        "space_complexity": "O(N^2) - input + output + block buffers",
        "huffman_overhead": "O(S log S) for codebook construction, S = unique symbols",
    }
    return analysis
