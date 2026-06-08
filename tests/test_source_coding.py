"""
Smoke tests for Project 2 Part A source coding.

Run from project root:
    python tests/test_source_coding.py
"""

from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import psnr
from src.source_coding import dct_2d, decode_image, encode_image, idct_2d


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_dct_idct_roundtrip():
    rng = np.random.default_rng(123)
    block = rng.normal(size=(8, 8))
    restored = idct_2d(dct_2d(block))
    assert_true(np.allclose(block, restored, atol=1e-9), "DCT/IDCT roundtrip failed")


def test_uniform_image_single_symbol_huffman():
    img = np.full((16, 16), 128, dtype=np.uint8)
    encoded = encode_image(img, quality=50)
    decoded = decode_image(encoded)
    assert_true(decoded.shape == img.shape, "Uniform image shape changed")
    assert_true(np.max(np.abs(decoded.astype(int) - img.astype(int))) == 0,
                "Uniform image should reconstruct exactly at DC level")


def test_non_multiple_of_8_image():
    rng = np.random.default_rng(456)
    img = rng.integers(0, 256, size=(17, 19), dtype=np.uint8)
    encoded = encode_image(img, quality=75)
    decoded = decode_image(encoded)
    assert_true(decoded.shape == img.shape, "Decoded image did not crop back to original size")
    assert_true(np.isfinite(psnr(img, decoded)), "PSNR should be finite for random lossy image")


def test_graph_dataset_first_image():
    graph_dir = PROJECT_ROOT / "graph"
    images = sorted(graph_dir.glob("*.png"))
    assert_true(len(images) > 0, "No graph images found")
    img = np.array(Image.open(images[0]).convert("L"))
    encoded = encode_image(img, quality=50)
    decoded = decode_image(encoded)
    score = psnr(img, decoded)
    assert_true(decoded.shape == img.shape, "Graph image shape changed")
    assert_true(score > 25.0, f"Graph image PSNR too low: {score:.2f} dB")
    assert_true(encoded.compression_ratio > 1.0,
                f"Expected compression ratio > 1, got {encoded.compression_ratio:.2f}")


def main():
    tests = [
        test_dct_idct_roundtrip,
        test_uniform_image_single_symbol_huffman,
        test_non_multiple_of_8_image,
        test_graph_dataset_first_image,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print("All source coding tests passed.")


if __name__ == "__main__":
    main()
