"""
Project 2 — Part A: Lossy Source Coding for Images
====================================================
DCT-based (JPEG-like) lossy image compression with Huffman entropy coding.

Evaluates: PSNR, Compression Ratio, Algorithm Complexity, Rate-Distortion.

Usage:
    python main.py                          # Run with test images in graph/
    python main.py --image path/to/img.png  # Run with a specific image
    python main.py --quality 75             # Specify quality factor (default: range test)
"""

import argparse
import sys
import os
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.source_coding import (
    encode_image, decode_image, encode_image_rgb, decode_image_rgb,
    measure_complexity, EncodedImage
)
from src.analysis import (
    evaluate_quality_range, plot_rate_distortion, plot_complexity,
    plot_compression_summary, print_analysis_table,
    complexity_analysis, psnr, mse
)


def load_image(path: str) -> np.ndarray:
    """Load an image and convert to grayscale numpy array."""
    img = Image.open(path).convert("L")
    return np.array(img)


def save_image(img: np.ndarray, path: str):
    """Save a numpy array as an image."""
    im = Image.fromarray(img.astype(np.uint8))
    im.save(path)


def demo_single_quality(img: np.ndarray, quality: int, output_dir: str):
    """Run encode/decode at a single quality and show results."""
    print(f"\n{'='*60}")
    print(f"  Source Coding Demo - Quality = {quality}")
    print(f"{'='*60}")
    print(f"  Quality parameter: {quality}")

    encoded = encode_image(img, quality)
    decoded = decode_image(encoded)
    p = psnr(img, decoded)

    print(f"  Image size    : {img.shape[1]}x{img.shape[0]}")
    print(f"  Original bits : {encoded.original_size_bits:,}")
    print(f"  Compressed    : {encoded.compressed_size_bits:,} bits ({encoded.compressed_size_bits/8:,} bytes)")
    print(f"  Compression   : {encoded.compression_ratio:.2f}x")
    print(f"  Bits/pixel    : {encoded.bpp:.4f}")
    print(f"  PSNR          : {p:.2f} dB")
    print(f"  MSE           : {mse(img, decoded):.4f}")

    # Save reconstructed image
    recon_path = os.path.join(output_dir, f"reconstructed_q{quality}.png")
    save_image(decoded, recon_path)
    print(f"  Reconstructed -> {recon_path}")

    # Save bitstream for channel coding (Part B)
    bs_path = os.path.join(output_dir, f"bitstream_q{quality}.bin")
    with open(bs_path, "wb") as f:
        f.write(encoded.bitstream)
    print(f"  Bitstream     -> {bs_path} ({len(encoded.bitstream)} bytes)")

    return encoded


def demo_range(img: np.ndarray, output_dir: str):
    """Run full evaluation across a range of quality factors."""
    print(f"\n{'='*60}")
    print(f"  Source Coding - Full Quality Range Evaluation")
    print(f"{'='*60}")
    qualities = [1, 5, 10, 25, 50, 75, 90, 100]
    print(f"  Image: {img.shape[1]}x{img.shape[0]}")
    print(f"  Quality parameters: {qualities}\n")

    results = evaluate_quality_range(img, qualities)

    print_analysis_table(results)

    # Generate plots
    plot_rate_distortion(results, os.path.join(output_dir, "rate_distortion.png"))
    plot_complexity(results, os.path.join(output_dir, "complexity.png"))
    plot_compression_summary(results, os.path.join(output_dir, "compression_summary.png"))

    # Complexity analysis
    print("\n--- Algorithm Complexity Analysis ---")
    comp = complexity_analysis(img)
    for k, v in comp.items():
        print(f"  {k}: {v}")

    # Save best-quality reconstruction
    best_q = max(results, key=lambda r: r["psnr_db"])
    encoded_best = encode_image(img, best_q["quality"])
    decoded_best = decode_image(encoded_best)
    save_image(decoded_best, os.path.join(output_dir, f"reconstructed_best_q{best_q['quality']}.png"))

    return results


def main():
    parser = argparse.ArgumentParser(description="Project 2 Part A: Lossy Source Coding")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a specific image file")
    parser.add_argument("--quality", type=int, default=None,
                        help="Single quality factor (1-100). If omitted, evaluates full range.")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--graph-dir", type=str, default="graph",
                        help="Directory with graph/image dataset (default: graph/)")
    args = parser.parse_args()

    if args.quality is not None and not (1 <= args.quality <= 100):
        parser.error("--quality must be in the range 1..100")

    os.makedirs(args.output, exist_ok=True)

    # Load image(s)
    if args.image:
        img_path = args.image
        if not os.path.exists(img_path):
            print(f"Error: image not found: {img_path}")
            sys.exit(1)
        print(f"Loading: {img_path}")
        img = load_image(img_path)
        images = [(os.path.basename(img_path), img)]
    else:
        # Load from graph/ directory
        graph_dir = args.graph_dir
        if not os.path.isdir(graph_dir):
            print(f"Error: graph directory not found: {graph_dir}")
            print("Please place test images in the 'graph/' directory or use --image.")
            sys.exit(1)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".pgm", ".ppm"}
        img_files = sorted(f for f in os.listdir(graph_dir)
                           if os.path.splitext(f)[1].lower() in exts)
        if not img_files:
            print(f"No image files found in '{graph_dir}/'. Supported: {exts}")
            print("Please add test images or use --image flag.")
            sys.exit(1)
        images = []
        for f in img_files:
            path = os.path.join(graph_dir, f)
            print(f"Loading: {path}")
            images.append((f, load_image(path)))

    # Process each image
    for name, img in images:
        print(f"\n{'#'*60}")
        print(f"# Processing: {name}")
        print(f"{'#'*60}")

        img_output_dir = os.path.join(args.output, os.path.splitext(name)[0])
        os.makedirs(img_output_dir, exist_ok=True)

        # Convert to grayscale for analysis (color images processed per-channel)
        if img.ndim == 3 and img.shape[2] == 3:
            gray = np.round(0.299 * img[:, :, 0].astype(np.float64)
                            + 0.587 * img[:, :, 1].astype(np.float64)
                            + 0.114 * img[:, :, 2].astype(np.float64)).astype(np.uint8)
        else:
            gray = img

        if args.quality is not None:
            demo_single_quality(gray, args.quality, img_output_dir)
        else:
            demo_range(gray, img_output_dir)

    print(f"\n{'='*60}")
    print(f"  Done! All results saved to: {args.output}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
