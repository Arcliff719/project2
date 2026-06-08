"""
Project 2 final integration experiment runner.

This script is prepared for the integration / analysis part of the project.
It batch-runs the complete pipeline:

    image -> lossy source coding -> channel coding -> BSC/BEC simulation
          -> channel decoding -> source decoding -> metric collection

Outputs:
    output_final/final_metrics.csv
    output_final/reconstructed_images/
    output_final/figures/psnr_vs_error_probability.png
    output_final/figures/ber_vs_error_probability.png
    output_final/figures/time_complexity_comparison.png

Example:
    python run_full_experiment.py --quality 50 --error-probs 0,0.01,0.05,0.1
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Make root imports work when this script is launched from the project folder.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.source_coding import encode_image, decode_image
from src.channel_coding import transmit_bytes
from src.analysis import psnr, mse

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm", ".ppm"}


def parse_list(value: str, allowed: set[str] | None = None) -> list[str]:
    items = [x.strip().lower() for x in value.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("at least one item is required")
    if allowed is not None:
        invalid = [x for x in items if x not in allowed]
        if invalid:
            raise argparse.ArgumentTypeError(f"invalid value(s): {invalid}; allowed: {sorted(allowed)}")
    return items


def parse_probabilities(value: str) -> list[float]:
    probs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        p = float(item)
        if not 0.0 <= p <= 1.0:
            raise argparse.ArgumentTypeError("probabilities must be in [0, 1]")
        probs.append(p)
    if not probs:
        raise argparse.ArgumentTypeError("at least one probability is required")
    return probs


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_grayscale_image(path: str) -> np.ndarray:
    """Load image as an 8-bit grayscale numpy array."""
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def save_grayscale_image(img: np.ndarray, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)


def discover_images(graph_dir: str, max_images: int | None = None) -> list[str]:
    if not os.path.isdir(graph_dir):
        raise FileNotFoundError(f"image directory not found: {graph_dir}")
    files = []
    for name in sorted(os.listdir(graph_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            files.append(os.path.join(graph_dir, name))
    if max_images is not None:
        files = files[:max_images]
    if not files:
        raise FileNotFoundError(f"no supported image files found in: {graph_dir}")
    return files


def safe_float(value) -> str:
    if value is None:
        return "nan"
    try:
        v = float(value)
    except Exception:
        return "nan"
    if math.isinf(v):
        return "inf"
    if math.isnan(v):
        return "nan"
    return f"{v:.8f}"


def run_experiment(args: argparse.Namespace) -> str:
    graph_dir = os.path.abspath(args.graph_dir)
    output_dir = os.path.abspath(args.output_dir)
    figures_dir = os.path.join(output_dir, "figures")
    reconstructed_dir = os.path.join(output_dir, "reconstructed_images")
    ensure_dir(output_dir)
    ensure_dir(figures_dir)
    ensure_dir(reconstructed_dir)

    channels = parse_list(args.channels, {"bsc", "bec"})
    codes = parse_list(args.codes, {"none", "repetition", "hamming"})
    error_probs = parse_probabilities(args.error_probs)
    images = discover_images(graph_dir, args.max_images)

    csv_path = os.path.join(output_dir, "final_metrics.csv")
    fieldnames = [
        "image", "width", "height", "quality", "channel", "code", "error_probability",
        "source_psnr_db", "recovered_psnr_db", "mse", "compression_ratio", "bpp",
        "source_bits", "compressed_bits", "coded_bits", "redundancy_rate",
        "channel_error_rate", "erasure_rate", "ber", "corrected_errors", "uncorrectable_blocks",
        "source_encode_time_s", "source_decode_time_s", "channel_encode_time_s",
        "channel_simulation_time_s", "channel_decode_time_s", "total_time_s", "status",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for image_index, image_path in enumerate(images):
            name = os.path.basename(image_path)
            stem = os.path.splitext(name)[0]
            print(f"Processing {name} ({image_index + 1}/{len(images)})...")

            original = load_grayscale_image(image_path)
            height, width = original.shape[:2]

            # Source encode / decode baseline.
            source_t0 = time.perf_counter()
            encoded = encode_image(original, args.quality)
            source_t1 = time.perf_counter()
            decoded_baseline = decode_image(encoded)
            source_t2 = time.perf_counter()
            source_psnr = psnr(original, decoded_baseline)

            baseline_path = os.path.join(reconstructed_dir, stem, f"baseline_q{args.quality}.png")
            save_grayscale_image(decoded_baseline, baseline_path)

            for channel in channels:
                for code in codes:
                    for p in error_probs:
                        seed = None if args.seed is None else args.seed + image_index * 100000 + int(p * 100000) + (0 if channel == "bsc" else 50000)
                        status = "ok"
                        recovered_psnr = float("nan")
                        recovered_mse = float("nan")

                        try:
                            recovered_stream, ch_result = transmit_bytes(
                                encoded.bitstream,
                                channel=channel,
                                code=code,
                                error_probability=p,
                                repetition_factor=args.repetition_factor,
                                seed=seed,
                            )
                        except Exception as exc:
                            status = f"channel_failed: {type(exc).__name__}: {exc}"
                            class EmptyResult:
                                source_bits = encoded.compressed_size_bits
                                coded_bits = 0
                                redundancy_rate = 0.0
                                channel_error_rate = float("nan")
                                erasure_rate = float("nan")
                                ber = float("nan")
                                corrected_errors = 0
                                uncorrectable_blocks = 0
                                encode_time_s = 0.0
                                channel_time_s = 0.0
                                decode_time_s = 0.0
                                total_time_s = 0.0
                            ch_result = EmptyResult()
                            recovered_stream = b""

                        if status == "ok":
                            try:
                                recovered_encoded = replace(encoded, bitstream=recovered_stream)
                                recovered_img = decode_image(recovered_encoded)
                                recovered_psnr = psnr(original, recovered_img)
                                recovered_mse = mse(original, recovered_img)

                                if args.save_images:
                                    p_tag = str(p).replace(".", "p")
                                    out_path = os.path.join(reconstructed_dir, stem, f"{channel}_{code}_p{p_tag}.png")
                                    save_grayscale_image(recovered_img, out_path)
                            except Exception as exc:
                                # Source decoder failure itself is useful evidence of insufficient channel protection.
                                # Channel metrics such as BER are still kept in the CSV row.
                                status = f"source_decode_failed: {type(exc).__name__}: {exc}"

                        writer.writerow({
                            "image": name,
                            "width": width,
                            "height": height,
                            "quality": args.quality,
                            "channel": channel,
                            "code": code,
                            "error_probability": p,
                            "source_psnr_db": safe_float(source_psnr),
                            "recovered_psnr_db": safe_float(recovered_psnr),
                            "mse": safe_float(recovered_mse),
                            "compression_ratio": safe_float(encoded.compression_ratio),
                            "bpp": safe_float(encoded.bpp),
                            "source_bits": encoded.original_size_bits,
                            "compressed_bits": encoded.compressed_size_bits,
                            "coded_bits": ch_result.coded_bits,
                            "redundancy_rate": safe_float(ch_result.redundancy_rate),
                            "channel_error_rate": safe_float(ch_result.channel_error_rate),
                            "erasure_rate": safe_float(ch_result.erasure_rate),
                            "ber": safe_float(ch_result.ber),
                            "corrected_errors": ch_result.corrected_errors,
                            "uncorrectable_blocks": ch_result.uncorrectable_blocks,
                            "source_encode_time_s": safe_float(source_t1 - source_t0),
                            "source_decode_time_s": safe_float(source_t2 - source_t1),
                            "channel_encode_time_s": safe_float(ch_result.encode_time_s),
                            "channel_simulation_time_s": safe_float(ch_result.channel_time_s),
                            "channel_decode_time_s": safe_float(ch_result.decode_time_s),
                            "total_time_s": safe_float((source_t1 - source_t0) + (source_t2 - source_t1) + ch_result.total_time_s),
                            "status": status,
                        })

    make_plots(csv_path, figures_dir)
    print(f"Metrics CSV saved to: {csv_path}")
    print(f"Figures saved to: {figures_dir}")
    return csv_path


def read_numeric_csv(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["error_probability", "recovered_psnr_db", "ber", "total_time_s"]:
                try:
                    row[key] = float(row[key])
                except Exception:
                    row[key] = float("nan")
            rows.append(row)
    return rows


def average_by(rows: Iterable[dict], y_key: str):
    grouped = defaultdict(list)
    for row in rows:
        y = row.get(y_key, float("nan"))
        p = row.get("error_probability", float("nan"))
        if math.isnan(y) or math.isnan(p):
            continue
        key = (row["channel"], row["code"], p)
        grouped[key].append(y)
    result = defaultdict(list)
    for (channel, code, p), values in grouped.items():
        result[(channel, code)].append((p, sum(values) / len(values)))
    for key in result:
        result[key].sort(key=lambda x: x[0])
    return result


def make_line_plot(grouped, title: str, ylabel: str, output_path: str):
    plt.figure(figsize=(8, 5))
    for (channel, code), values in sorted(grouped.items()):
        xs = [x for x, _ in values]
        ys = [y for _, y in values]
        plt.plot(xs, ys, marker="o", label=f"{channel.upper()} + {code}")
    plt.xlabel("Channel error / erasure probability")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_time_plot(rows: list[dict], output_path: str):
    grouped = defaultdict(list)
    for row in rows:
        t = row.get("total_time_s", float("nan"))
        if not math.isnan(t):
            grouped[f"{row['channel'].upper()}+{row['code']}"].append(t)
    labels = []
    means = []
    for key in sorted(grouped):
        labels.append(key)
        values = grouped[key]
        means.append(sum(values) / len(values))
    plt.figure(figsize=(9, 5))
    plt.bar(labels, means)
    plt.xlabel("Channel model and channel code")
    plt.ylabel("Average total time (s)")
    plt.title("Algorithm Complexity Comparison")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_plots(csv_path: str, figures_dir: str) -> None:
    rows = read_numeric_csv(csv_path)
    make_line_plot(
        average_by(rows, "recovered_psnr_db"),
        "Recovered Image Quality vs Channel Error Probability",
        "Average recovered PSNR (dB)",
        os.path.join(figures_dir, "psnr_vs_error_probability.png"),
    )
    make_line_plot(
        average_by(rows, "ber"),
        "Bit Error Rate vs Channel Error Probability",
        "Average BER after channel decoding",
        os.path.join(figures_dir, "ber_vs_error_probability.png"),
    )
    make_time_plot(rows, os.path.join(figures_dir, "time_complexity_comparison.png"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Project 2 final integrated experiments.")
    parser.add_argument("--graph-dir", default="graph", help="input image dataset folder")
    parser.add_argument("--output-dir", default="output_final", help="folder for final metrics and figures")
    parser.add_argument("--quality", type=int, default=50, help="DCT/JPEG-like quality factor, 1-100")
    parser.add_argument("--channels", default="bsc,bec", help="comma-separated channels: bsc,bec")
    parser.add_argument("--codes", default="none,repetition,hamming", help="comma-separated channel codes: none,repetition,hamming")
    parser.add_argument("--error-probs", default="0,0.01,0.05,0.1", help="comma-separated probabilities in [0,1]")
    parser.add_argument("--repetition-factor", type=int, default=3, help="odd repetition factor")
    parser.add_argument("--seed", type=int, default=2026, help="random seed; use --seed -1 for random")
    parser.add_argument("--max-images", type=int, default=None, help="optional limit for quick tests")
    parser.add_argument("--save-images", action="store_true", help="save every channel-recovered image")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be in the range 1..100")
    if args.repetition_factor < 1 or args.repetition_factor % 2 == 0:
        parser.error("--repetition-factor must be an odd positive integer")
    if args.seed == -1:
        args.seed = None
    run_experiment(args)


if __name__ == "__main__":
    main()
