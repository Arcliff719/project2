import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.channel_coding import (  # noqa: E402
    bits_to_bytes,
    bytes_to_bits,
    decode_hamming74,
    encode_hamming74,
    simulate_bec,
    simulate_bsc,
    transmit_bytes,
)
from src.source_coding import decode_image, encode_image  # noqa: E402
from src.analysis import psnr  # noqa: E402


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_byte_bit_roundtrip():
    data = bytes([0, 1, 127, 128, 255])
    bits = bytes_to_bits(data)
    assert_true(bits.dtype == np.uint8, "bits should be uint8")
    assert_true(bits_to_bytes(bits) == data, "byte/bit conversion changed the payload")


def test_hamming_corrects_single_bsc_error_per_word():
    bits = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
    coded, _ = encode_hamming74(bits)
    corrupted = coded.astype(np.int16)
    corrupted[0] ^= 1
    corrupted[10] ^= 1
    decoded, corrected, uncorrectable = decode_hamming74(corrupted, len(bits))
    assert_true(np.array_equal(decoded, bits), "Hamming(7,4) failed to correct one error per codeword")
    assert_true(corrected == 2, f"expected 2 corrected words, got {corrected}")
    assert_true(uncorrectable == 0, "single-error Hamming decode should be correctable")


def test_hamming_recovers_single_bec_erasure_per_word():
    bits = np.array([1, 1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
    coded, _ = encode_hamming74(bits)
    erased = coded.astype(np.int16)
    erased[3] = -1
    erased[12] = -1
    decoded, corrected, uncorrectable = decode_hamming74(erased, len(bits))
    assert_true(np.array_equal(decoded, bits), "Hamming(7,4) failed to recover one erasure per codeword")
    assert_true(corrected == 2, f"expected 2 recovered erasures, got {corrected}")
    assert_true(uncorrectable == 0, "single-erasure Hamming decode should be correctable")


def test_channel_models_are_reproducible():
    bits = np.ones(64, dtype=np.uint8)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    bsc1, errors1, _ = simulate_bsc(bits, 0.25, rng1)
    bsc2, errors2, _ = simulate_bsc(bits, 0.25, rng2)
    assert_true(np.array_equal(bsc1, bsc2) and errors1 == errors2, "BSC simulation should be seed-reproducible")

    rng3 = np.random.default_rng(456)
    rng4 = np.random.default_rng(456)
    bec1, _, erasures1 = simulate_bec(bits, 0.25, rng3)
    bec2, _, erasures2 = simulate_bec(bits, 0.25, rng4)
    assert_true(np.array_equal(bec1, bec2) and erasures1 == erasures2, "BEC simulation should be seed-reproducible")


def test_transmit_bytes_zero_noise_roundtrip():
    data = os.urandom(32)
    for channel in ("bsc", "bec"):
        recovered, result = transmit_bytes(data, channel=channel, code="hamming", error_probability=0.0, seed=7)
        assert_true(recovered == data, f"zero-noise {channel} transmission changed payload")
        assert_true(result.ber == 0.0, f"zero-noise {channel} BER should be zero")


def test_source_bitstream_through_hamming_bec_zero_noise():
    graph_dir = PROJECT_ROOT / "graph"
    images = sorted(graph_dir.glob("*.png"))
    assert_true(len(images) > 0, "No graph images found")
    img = np.array(Image.open(images[0]).convert("L"))
    encoded = encode_image(img, quality=50)
    recovered_stream, result = transmit_bytes(encoded.bitstream, channel="bec", code="hamming", error_probability=0.0, seed=42)
    encoded.bitstream = recovered_stream
    decoded = decode_image(encoded)
    assert_true(result.ber == 0.0, "Protected zero-noise source bitstream should have BER=0")
    assert_true(psnr(img, decoded) > 25.0, "Recovered graph PSNR is unexpectedly low after zero-noise channel")


def main():
    tests = [
        test_byte_bit_roundtrip,
        test_hamming_corrects_single_bsc_error_per_word,
        test_hamming_recovers_single_bec_erasure_per_word,
        test_channel_models_are_reproducible,
        test_transmit_bytes_zero_noise_roundtrip,
        test_source_bitstream_through_hamming_bec_zero_noise,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print("All channel coding tests passed.")


if __name__ == "__main__":
    main()
