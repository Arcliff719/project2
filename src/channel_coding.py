"""
Channel coding and channel simulation for Project 2 Part B.

This module converts the compressed source-coding byte stream into bits,
protects it with a selectable channel code, transmits it through simulated
BSC/BEC channels, and reconstructs a byte stream for the source decoder.
"""

from dataclasses import dataclass
import time
from typing import Iterable, Literal, Tuple

import numpy as np

ChannelName = Literal["bsc", "bec"]
CodeName = Literal["none", "repetition", "hamming"]


@dataclass
class ChannelResult:
    """Metrics collected for one coded transmission experiment."""

    channel: str
    code: str
    error_probability: float
    source_bits: int
    coded_bits: int
    decoded_bits: int
    channel_bit_errors: int
    channel_erasures: int
    information_bit_errors: int
    corrected_errors: int
    uncorrectable_blocks: int
    encode_time_s: float
    channel_time_s: float
    decode_time_s: float

    @property
    def redundancy_rate(self) -> float:
        """Coded bits per source bit."""
        return self.coded_bits / self.source_bits if self.source_bits else 0.0

    @property
    def channel_error_rate(self) -> float:
        """Raw channel flip rate over coded bits for BSC experiments."""
        return self.channel_bit_errors / self.coded_bits if self.coded_bits else 0.0

    @property
    def erasure_rate(self) -> float:
        """Raw erasure rate over coded bits for BEC experiments."""
        return self.channel_erasures / self.coded_bits if self.coded_bits else 0.0

    @property
    def ber(self) -> float:
        """End-to-end information-bit error rate after channel decoding."""
        return self.information_bit_errors / self.source_bits if self.source_bits else 0.0

    @property
    def total_time_s(self) -> float:
        return self.encode_time_s + self.channel_time_s + self.decode_time_s


# ============================================================
# Bit/byte conversion
# ============================================================


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert bytes to a uint8 vector of 0/1 bits (MSB first)."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes-like")
    if len(data) == 0:
        return np.zeros(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))


def bits_to_bytes(bits: Iterable[int]) -> bytes:
    """Pack a 0/1 bit vector into bytes (MSB first)."""
    arr = np.asarray(list(bits) if not isinstance(bits, np.ndarray) else bits, dtype=np.uint8)
    if arr.ndim != 1:
        raise ValueError("bits must be a one-dimensional sequence")
    if arr.size % 8 != 0:
        raise ValueError(f"bit length must be a multiple of 8, got {arr.size}")
    if arr.size == 0:
        return b""
    return np.packbits(arr).tobytes()


# ============================================================
# Channel encoders / decoders
# ============================================================


def encode_none(bits: np.ndarray) -> Tuple[np.ndarray, int]:
    return np.asarray(bits, dtype=np.uint8).copy(), 0


def decode_none(received: np.ndarray, original_length: int) -> Tuple[np.ndarray, int, int]:
    arr = np.asarray(received).copy()
    erasures = arr < 0
    arr[erasures] = 0
    return arr.astype(np.uint8)[:original_length], 0, int(np.count_nonzero(erasures))


def encode_repetition(bits: np.ndarray, factor: int = 3) -> Tuple[np.ndarray, int]:
    """Encode by repeating each bit an odd number of times."""
    if factor < 1 or factor % 2 == 0:
        raise ValueError("repetition factor must be an odd positive integer")
    return np.repeat(np.asarray(bits, dtype=np.uint8), factor), 0


def decode_repetition(received: np.ndarray, original_length: int, factor: int = 3) -> Tuple[np.ndarray, int, int]:
    """Majority-vote repetition decoder; erasures are ignored when present."""
    if factor < 1 or factor % 2 == 0:
        raise ValueError("repetition factor must be an odd positive integer")
    arr = np.asarray(received, dtype=np.int16)
    n_groups = int(np.ceil(original_length / 1.0))
    needed = n_groups * factor
    if arr.size < needed:
        arr = np.pad(arr, (0, needed - arr.size), constant_values=-1)
    arr = arr[:needed].reshape((-1, factor))

    decoded = np.zeros(arr.shape[0], dtype=np.uint8)
    corrected = 0
    uncorrectable = 0
    for i, group in enumerate(arr):
        known = group[group >= 0]
        if known.size == 0:
            decoded[i] = 0
            uncorrectable += 1
            continue
        ones = int(np.count_nonzero(known == 1))
        zeros = int(np.count_nonzero(known == 0))
        decoded[i] = 1 if ones > zeros else 0
        corrected += int(np.count_nonzero(known != decoded[i]) + np.count_nonzero(group < 0))
    return decoded[:original_length], corrected, uncorrectable


def _hamming74_encode_nibble(nibble: np.ndarray) -> np.ndarray:
    d1, d2, d3, d4 = [int(x) for x in nibble]
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p4 = d2 ^ d3 ^ d4
    return np.array([p1, p2, d1, p4, d2, d3, d4], dtype=np.uint8)


def encode_hamming74(bits: np.ndarray) -> Tuple[np.ndarray, int]:
    """Encode bits with systematic Hamming(7,4); returns coded bits and pad length."""
    arr = np.asarray(bits, dtype=np.uint8)
    pad = (4 - arr.size % 4) % 4
    if pad:
        arr = np.pad(arr, (0, pad), constant_values=0)
    blocks = arr.reshape((-1, 4)) if arr.size else np.zeros((0, 4), dtype=np.uint8)
    encoded = np.concatenate([_hamming74_encode_nibble(block) for block in blocks]) if len(blocks) else np.zeros(0, dtype=np.uint8)
    return encoded, pad


def _hamming74_syndrome(codeword: np.ndarray) -> int:
    c = [0] + [int(x) for x in codeword]
    s1 = c[1] ^ c[3] ^ c[5] ^ c[7]
    s2 = c[2] ^ c[3] ^ c[6] ^ c[7]
    s4 = c[4] ^ c[5] ^ c[6] ^ c[7]
    return s1 + 2 * s2 + 4 * s4


def _hamming74_decode_word(word: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Decode one Hamming(7,4) word; returns data, corrected_count, uncorrectable_count."""
    arr = np.asarray(word, dtype=np.int16).copy()
    erasure_positions = np.flatnonzero(arr < 0)

    if erasure_positions.size == 1:
        # Try both values and keep the one that satisfies all parity checks.
        pos = int(erasure_positions[0])
        for value in (0, 1):
            trial = arr.copy()
            trial[pos] = value
            if _hamming74_syndrome(trial.astype(np.uint8)) == 0:
                return trial[[2, 4, 5, 6]].astype(np.uint8), 1, 0
        arr[pos] = 0
        return arr[[2, 4, 5, 6]].astype(np.uint8), 0, 1

    if erasure_positions.size > 1:
        arr[erasure_positions] = 0
        return arr[[2, 4, 5, 6]].astype(np.uint8), 0, 1

    syndrome = _hamming74_syndrome(arr.astype(np.uint8))
    corrected = 0
    if syndrome:
        arr[syndrome - 1] ^= 1
        corrected = 1
    return arr[[2, 4, 5, 6]].astype(np.uint8), corrected, 0


def decode_hamming74(received: np.ndarray, original_length: int) -> Tuple[np.ndarray, int, int]:
    """Decode Hamming(7,4) blocks and trim to the source bit length."""
    arr = np.asarray(received, dtype=np.int16)
    if arr.size % 7:
        arr = np.pad(arr, (0, 7 - arr.size % 7), constant_values=-1)
    decoded_blocks = []
    corrected = 0
    uncorrectable = 0
    for word in arr.reshape((-1, 7)):
        data, corr, uncorr = _hamming74_decode_word(word)
        decoded_blocks.append(data)
        corrected += corr
        uncorrectable += uncorr
    decoded = np.concatenate(decoded_blocks) if decoded_blocks else np.zeros(0, dtype=np.uint8)
    return decoded[:original_length], corrected, uncorrectable


# ============================================================
# Channel models
# ============================================================


def simulate_bsc(bits: np.ndarray, error_probability: float, rng: np.random.Generator) -> Tuple[np.ndarray, int, int]:
    """Binary Symmetric Channel: each coded bit flips independently with probability p."""
    if not 0.0 <= error_probability <= 1.0:
        raise ValueError("error_probability must be in [0, 1]")
    arr = np.asarray(bits, dtype=np.uint8)
    flips = rng.random(arr.size) < error_probability
    received = arr.copy()
    received[flips] ^= 1
    return received.astype(np.int16), int(np.count_nonzero(flips)), 0


def simulate_bec(bits: np.ndarray, erasure_probability: float, rng: np.random.Generator) -> Tuple[np.ndarray, int, int]:
    """Binary Erasure Channel: each coded bit is replaced by -1 with probability p."""
    if not 0.0 <= erasure_probability <= 1.0:
        raise ValueError("erasure_probability must be in [0, 1]")
    arr = np.asarray(bits, dtype=np.int16).copy()
    erasures = rng.random(arr.size) < erasure_probability
    arr[erasures] = -1
    return arr, 0, int(np.count_nonzero(erasures))


# ============================================================
# End-to-end transmission helper
# ============================================================


def transmit_bytes(
    data: bytes,
    channel: ChannelName = "bsc",
    code: CodeName = "hamming",
    error_probability: float = 0.05,
    repetition_factor: int = 3,
    seed: int | None = None,
) -> Tuple[bytes, ChannelResult]:
    """
    Encode, transmit, and channel-decode a compressed byte stream.

    Returns:
        (recovered_bytes, ChannelResult) where recovered_bytes has the same
        length as the input byte stream so it can be passed back to Part A's
        source decoder.
    """
    source_bits = bytes_to_bits(data)
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    if code == "none":
        coded_bits, _ = encode_none(source_bits)
    elif code == "repetition":
        coded_bits, _ = encode_repetition(source_bits, repetition_factor)
    elif code == "hamming":
        coded_bits, _ = encode_hamming74(source_bits)
    else:
        raise ValueError(f"Unsupported channel code: {code}")
    t1 = time.perf_counter()

    if channel == "bsc":
        received, channel_errors, erasures = simulate_bsc(coded_bits, error_probability, rng)
    elif channel == "bec":
        received, channel_errors, erasures = simulate_bec(coded_bits, error_probability, rng)
    else:
        raise ValueError(f"Unsupported channel: {channel}")
    t2 = time.perf_counter()

    if code == "none":
        decoded_bits, corrected, uncorrectable = decode_none(received, source_bits.size)
    elif code == "repetition":
        decoded_bits, corrected, uncorrectable = decode_repetition(received, source_bits.size, repetition_factor)
    else:
        decoded_bits, corrected, uncorrectable = decode_hamming74(received, source_bits.size)
    t3 = time.perf_counter()

    information_errors = int(np.count_nonzero(decoded_bits[:source_bits.size] != source_bits))
    recovered = bits_to_bytes(decoded_bits[:source_bits.size])
    result = ChannelResult(
        channel=channel,
        code=code,
        error_probability=error_probability,
        source_bits=int(source_bits.size),
        coded_bits=int(coded_bits.size),
        decoded_bits=int(decoded_bits.size),
        channel_bit_errors=channel_errors,
        channel_erasures=erasures,
        information_bit_errors=information_errors,
        corrected_errors=corrected,
        uncorrectable_blocks=uncorrectable,
        encode_time_s=t1 - t0,
        channel_time_s=t2 - t1,
        decode_time_s=t3 - t2,
    )
    return recovered, result
