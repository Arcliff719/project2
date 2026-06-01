"""
Lossy Source Coding for Images — DCT-based (JPEG-like) compression.
Supports configurable quality factor, Huffman entropy coding, and
bitstream generation for transmission over BSC/BEC channels.
"""

import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Tuple, List
import heapq
import time
import struct


# ============================================================
# 8x8 DCT / IDCT
# ============================================================

def _build_dct_matrix(n: int = 8) -> np.ndarray:
    """Build the orthonormal DCT-II transform matrix used by JPEG blocks."""
    mat = np.zeros((n, n), dtype=np.float64)
    for u in range(n):
        cu = 1.0 / np.sqrt(2) if u == 0 else 1.0
        for x in range(n):
            mat[u, x] = 0.5 * cu * np.cos((2 * x + 1) * u * np.pi / (2 * n))
    return mat


_DCT8 = _build_dct_matrix(8)

def dct_2d(block: np.ndarray) -> np.ndarray:
    """Apply 2D DCT-II to an 8x8 block."""
    if block.shape != (8, 8):
        raise ValueError(f"DCT block must be 8x8, got {block.shape}")
    return _DCT8 @ block.astype(np.float64) @ _DCT8.T


def idct_2d(block: np.ndarray) -> np.ndarray:
    """Apply 2D IDCT-II to an 8x8 block."""
    if block.shape != (8, 8):
        raise ValueError(f"IDCT block must be 8x8, got {block.shape}")
    return _DCT8.T @ block.astype(np.float64) @ _DCT8


# ============================================================
# Quantization
# ============================================================

# Standard JPEG luminance quantization matrix
_BASE_Q = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61 ],
    [12, 12, 14, 19, 26,  58,  60,  55 ],
    [14, 13, 16, 24, 40,  57,  69,  56 ],
    [14, 17, 22, 29, 51,  87,  80,  62 ],
    [18, 22, 37, 56, 68,  109, 103, 77 ],
    [24, 35, 55, 64, 81,  104, 113, 92 ],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99 ]
], dtype=np.float64)


def make_quant_table(quality: int = 50) -> np.ndarray:
    """
    Build a quantization table scaled by quality factor (1–100).
    quality=100 → near-lossless; quality=1 → maximum compression.
    Uses the IJG JPEG scaling formula.
    """
    quality = max(1, min(100, quality))
    if quality < 50:
        scale = 5000.0 / quality
    else:
        scale = 200.0 - 2.0 * quality
    qt = np.floor((_BASE_Q * scale + 50.0) / 100.0)
    qt = np.clip(qt, 1.0, 255.0)
    return qt


def quantize(dct_block: np.ndarray, qt: np.ndarray) -> np.ndarray:
    """Quantize DCT coefficients (element-wise division + rounding)."""
    return np.round(dct_block / qt).astype(np.int32)


def dequantize(qblock: np.ndarray, qt: np.ndarray) -> np.ndarray:
    """Dequantize (element-wise multiplication)."""
    return qblock.astype(np.float64) * qt


# ============================================================
# Zigzag scan
# ============================================================

_ZIGZAG = [
    0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
]

_ZIGZAG_INV = [0] * 64
for _i, _v in enumerate(_ZIGZAG):
    _ZIGZAG_INV[_v] = _i


def zigzag(block: np.ndarray) -> np.ndarray:
    flat = block.flatten()
    return np.array([flat[i] for i in _ZIGZAG], dtype=np.int32)


def izigzag(arr: np.ndarray) -> np.ndarray:
    block = np.zeros(64, dtype=np.int32)
    for i, v in enumerate(arr):
        block[_ZIGZAG[i]] = v
    return block.reshape((8, 8))


# ============================================================
# DPCM + RLE for DCT coefficients
# ============================================================

def encode_dc_coeffs(dc_seq: List[int]) -> Tuple[List[int], List[int]]:
    """
    Differential Pulse Code Modulation (DPCM) for DC coefficients.
    Returns (categories, values) for Huffman coding.
    """
    diffs = [dc_seq[0]]
    for i in range(1, len(dc_seq)):
        diffs.append(dc_seq[i] - dc_seq[i - 1])
    return diffs


def decode_dc_coeffs(dc_diffs: List[int]) -> List[int]:
    dc_seq = [dc_diffs[0]]
    for i in range(1, len(dc_diffs)):
        dc_seq.append(dc_seq[-1] + dc_diffs[i])
    return dc_seq


def category_of(val: int) -> int:
    """SSSS = number of bits needed to represent |val|."""
    if val == 0:
        return 0
    return int(np.floor(np.log2(abs(val))) + 1)


def encode_ac_runlength(ac_coeffs: np.ndarray) -> List[Tuple[int, int]]:
    """
    Run-length encode AC coefficients (zigzag order, excluding DC).
    Returns list of (run_length, value) tuples, terminated by (0, 0) EOB.
    """
    symbols = []
    run = 0
    for coeff in ac_coeffs:
        if coeff == 0:
            run += 1
            if run >= 16:  # ZRL marker
                symbols.append((15, 0))
                run = 0
        else:
            while run >= 16:
                symbols.append((15, 0))
                run -= 16
            symbols.append((run, int(coeff)))
            run = 0
    if run > 0:
        symbols.append((0, 0))  # EOB
    else:
        symbols.append((0, 0))
    return symbols


def decode_ac_runlength(symbols: List[Tuple[int, int]]) -> List[int]:
    coeffs = []
    for run, val in symbols:
        if run == 0 and val == 0:
            break
        coeffs.extend([0] * run)
        coeffs.append(val)
    # Pad to 63
    coeffs.extend([0] * (63 - len(coeffs)))
    return coeffs[:63]


# ============================================================
# Huffman coding
# ============================================================

class HuffmanNode:
    __slots__ = ('symbol', 'freq', 'left', 'right')

    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freq_map: dict) -> HuffmanNode:
    heap = [HuffmanNode(sym, f) for sym, f in freq_map.items()]
    heapq.heapify(heap)
    if len(heap) == 0:
        return None
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, HuffmanNode(freq=a.freq + b.freq, left=a, right=b))
    return heap[0]


def build_huffman_codes(root: HuffmanNode) -> dict:
    codes = {}

    def _traverse(node, code):
        if node is None:
            return
        if node.left is None and node.right is None:
            codes[node.symbol] = code if code else "0"
            return
        _traverse(node.left, code + "0")
        _traverse(node.right, code + "1")

    _traverse(root, "")
    return codes


def huffman_encode(symbols: list, codes: dict) -> str:
    return "".join(codes[s] for s in symbols)


def huffman_decode(bitstr: str, root: HuffmanNode) -> list:
    if root is None:
        return []
    if root.left is None and root.right is None:
        return [root.symbol] * len(bitstr)
    decoded = []
    node = root
    for b in bitstr:
        node = node.left if b == "0" else node.right
        if node is None:
            raise ValueError("Invalid Huffman bitstream for the supplied tree")
        if node.left is None and node.right is None:
            decoded.append(node.symbol)
            node = root
    return decoded


# ============================================================
# Bitstream packing
# ============================================================

def pack_bitstream(bitstr: str) -> bytes:
    """Pack a binary string into bytes (MSB first)."""
    # Pad to multiple of 8
    pad = (8 - len(bitstr) % 8) % 8
    bitstr += "0" * pad
    data = bytearray()
    data.append(pad)  # First byte = padding count
    for i in range(0, len(bitstr), 8):
        data.append(int(bitstr[i:i + 8], 2))
    return bytes(data)


def unpack_bitstream(data: bytes) -> str:
    """Unpack bytes back to binary string."""
    if not data:
        raise ValueError("Empty bitstream")
    pad = data[0]
    if pad > 7:
        raise ValueError(f"Invalid bitstream padding length: {pad}")
    bits = "".join(f"{b:08b}" for b in data[1:])
    if pad > 0:
        bits = bits[:-pad]
    return bits


# ============================================================
# Image block processing
# ============================================================

def pad_image(img: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Pad image dimensions to multiples of 8."""
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {img.shape}")
    if img.size == 0:
        raise ValueError("Image must not be empty")
    h, w = img.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")
    return img, h, w


def blocks_from_image(img: np.ndarray) -> np.ndarray:
    """Split image into 8x8 blocks. Returns array of shape (n_blocks, 8, 8)."""
    h, w = img.shape
    blocks = []
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            blocks.append(img[i:i + 8, j:j + 8].astype(np.float64) - 128.0)
    return np.array(blocks)


def image_from_blocks(blocks: np.ndarray, h: int, w: int) -> np.ndarray:
    """Reconstruct image from 8x8 blocks."""
    padded_h = h + (8 - h % 8) % 8
    padded_w = w + (8 - w % 8) % 8
    img = np.zeros((padded_h, padded_w), dtype=np.float64)
    blk_idx = 0
    for i in range(0, padded_h, 8):
        for j in range(0, padded_w, 8):
            img[i:i + 8, j:j + 8] = blocks[blk_idx] + 128.0
            blk_idx += 1
    return np.clip(np.round(img), 0, 255).astype(np.uint8)[:h, :w]


# ============================================================
# Main encoder / decoder
# ============================================================

@dataclass
class EncodedImage:
    """Container for encoded image data ready for channel transmission."""
    bitstream: bytes          # Compressed bitstream
    img_shape: Tuple[int, int]  # Original (height, width)
    quality: int
    quant_table: np.ndarray
    dc_huff_root: HuffmanNode
    ac_huff_root: HuffmanNode
    # Statistics
    original_size_bits: int
    compressed_size_bits: int

    @property
    def compression_ratio(self) -> float:
        return self.original_size_bits / self.compressed_size_bits

    @property
    def bpp(self) -> float:
        """Bits per pixel."""
        h, w = self.img_shape
        return self.compressed_size_bits / (h * w)


def encode_image(img: np.ndarray, quality: int = 50) -> EncodedImage:
    """
    Lossy source encode a grayscale image.

    Pipeline: DCT → Quantize → Zigzag → DPCM(DC) + RLE(AC) → Huffman → Bitstream

    Args:
        img: 2D uint8 grayscale image (H, W)
        quality: 1–100, higher = better quality

    Returns:
        EncodedImage with compressed bitstream and metadata.
    """
    t_start = time.perf_counter()

    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy.ndarray")

    if img.ndim == 3:
        # Convert RGB/RGBA to grayscale using standard luminance weights
        if img.shape[2] < 3:
            raise ValueError(f"Expected at least 3 channels for color image, got shape {img.shape}")
        img = np.round(0.299 * img[:, :, 0].astype(np.float64)
                       + 0.587 * img[:, :, 1].astype(np.float64)
                       + 0.114 * img[:, :, 2].astype(np.float64)).astype(np.uint8)
    elif img.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale or 3D color image, got shape {img.shape}")

    if img.size == 0:
        raise ValueError("Image must not be empty")
    img = np.clip(img, 0, 255).astype(np.uint8, copy=False)

    orig_h, orig_w = img.shape
    original_bits = orig_h * orig_w * 8

    # Pad and split
    padded, _, _ = pad_image(img)
    blocks = blocks_from_image(padded)
    qt = make_quant_table(quality)

    # Per-block encode
    dc_diffs = []
    ac_symbols_all = []

    for blk in blocks:
        dct_blk = dct_2d(blk)
        qblk = quantize(dct_blk, qt)
        zig = zigzag(qblk)
        # DC (index 0)
        dc_diffs.append(int(zig[0]))
        # AC (indices 1..63)
        ac_syms = encode_ac_runlength(zig[1:])
        ac_symbols_all.append(ac_syms)

    # DPCM on DC
    dc_diffs_enc = encode_dc_coeffs(dc_diffs)

    # Build Huffman tables
    # DC: encode (category, value) pairs
    dc_categories = [category_of(d) for d in dc_diffs_enc]
    dc_freq = Counter(dc_categories)
    dc_huff_root = build_huffman_tree(dc_freq)
    dc_huff_codes = build_huffman_codes(dc_huff_root)

    # AC: encode (run, category) pairs — category of the non-zero value
    ac_rs_symbols = []  # (run, ssss)
    ac_values = []      # actual coefficient values
    for syms in ac_symbols_all:
        for run, val in syms:
            if run == 0 and val == 0:
                ac_rs_symbols.append((0, 0))  # EOB
                break
            ssss = category_of(val)
            ac_rs_symbols.append((run, ssss))
            ac_values.append(val)

    ac_freq = Counter(ac_rs_symbols)
    ac_huff_root = build_huffman_tree(ac_freq)
    ac_huff_codes = build_huffman_codes(ac_huff_root)

    # Helper: encode signed integer as category + magnitude bits
    def _encode_signed(val: int) -> str:
        cat = category_of(val)
        if cat == 0:
            return ""
        if val > 0:
            return format(val, f"0{cat}b")
        else:
            return format((1 << cat) - 1 + val, f"0{cat}b")

    # --- DC section (Huffman bits + value bits, stored separately) ---
    dc_huff_bits = huffman_encode(dc_categories, dc_huff_codes)
    dc_val_bits = "".join(_encode_signed(d) for d in dc_diffs_enc)

    # --- AC section (Huffman bits + value bits, stored separately) ---
    ac_huff_bits = huffman_encode(ac_rs_symbols, ac_huff_codes)
    ac_val_bits = "".join(_encode_signed(val) for val in ac_values)

    # Combine with explicit section lengths
    # [img_h:16][img_w:16][quality:8][n_blocks:32]
    # [dc_huff_len:32][dc_val_len:32] [dc_huff_bits] [dc_val_bits]
    # [ac_huff_len:32][ac_val_len:32] [ac_huff_bits] [ac_val_bits]
    bitstr = ""
    bitstr += format(orig_h, "016b")
    bitstr += format(orig_w, "016b")
    bitstr += format(quality, "08b")
    bitstr += format(len(blocks), "032b")
    bitstr += format(len(dc_huff_bits), "032b")
    bitstr += format(len(dc_val_bits), "032b")
    bitstr += dc_huff_bits
    bitstr += dc_val_bits
    bitstr += format(len(ac_huff_bits), "032b")
    bitstr += format(len(ac_val_bits), "032b")
    bitstr += ac_huff_bits
    bitstr += ac_val_bits

    packed = pack_bitstream(bitstr)
    compressed_bits = len(packed) * 8
    elapsed = time.perf_counter() - t_start

    return EncodedImage(
        bitstream=packed,
        img_shape=(orig_h, orig_w),
        quality=quality,
        quant_table=qt,
        dc_huff_root=dc_huff_root,
        ac_huff_root=ac_huff_root,
        original_size_bits=original_bits,
        compressed_size_bits=compressed_bits,
    )


def decode_image(encoded: EncodedImage) -> np.ndarray:
    """
    Decode an EncodedImage back to a reconstructed image.

    Pipeline: Bitstream → Huffman decode → DPCM⁻¹ + RLE⁻¹ → IZigzag → Dequantize → IDCT
    """
    bitstr = unpack_bitstream(encoded.bitstream)

    # Parse header
    pos = 0
    orig_h = int(bitstr[pos:pos + 16], 2); pos += 16
    orig_w = int(bitstr[pos:pos + 16], 2); pos += 16
    quality = int(bitstr[pos:pos + 8], 2); pos += 8
    n_blocks = int(bitstr[pos:pos + 32], 2); pos += 32

    dc_huff_len = int(bitstr[pos:pos + 32], 2); pos += 32
    dc_val_len = int(bitstr[pos:pos + 32], 2); pos += 32

    dc_huff_bits = bitstr[pos:pos + dc_huff_len]; pos += dc_huff_len
    dc_val_bits = bitstr[pos:pos + dc_val_len]; pos += dc_val_len

    ac_huff_len = int(bitstr[pos:pos + 32], 2); pos += 32
    ac_val_len = int(bitstr[pos:pos + 32], 2); pos += 32

    ac_huff_bits = bitstr[pos:pos + ac_huff_len]; pos += ac_huff_len
    ac_val_bits = bitstr[pos:pos + ac_val_len]; pos += ac_val_len

    # Decode DC categories from Huffman bits
    dc_categories = huffman_decode(dc_huff_bits, encoded.dc_huff_root)
    if len(dc_categories) != n_blocks:
        raise ValueError(f"Decoded {len(dc_categories)} DC symbols, expected {n_blocks}")
    # Decode DC values from raw value bits
    dc_diffs = []
    vpos = 0
    for cat in dc_categories:
        if cat == 0:
            dc_diffs.append(0)
        else:
            bits = dc_val_bits[vpos:vpos + cat]
            vpos += cat
            val = int(bits, 2)
            if val < (1 << (cat - 1)):  # negative
                val = val - (1 << cat) + 1
            dc_diffs.append(val)
    dc_coeffs = decode_dc_coeffs(dc_diffs)
    if len(dc_coeffs) != n_blocks:
        raise ValueError(f"Decoded {len(dc_coeffs)} DC coefficients, expected {n_blocks}")

    # Decode AC run/size symbols from Huffman bits
    ac_rs_symbols = huffman_decode(ac_huff_bits, encoded.ac_huff_root)
    # Decode AC values from raw value bits (process ALL symbols, not stopping at EOB)
    ac_values = []
    vpos = 0
    for run, ssss in ac_rs_symbols:
        if run == 0 and ssss == 0:
            continue  # EOB has no associated value
        if ssss > 0:
            bits = ac_val_bits[vpos:vpos + ssss]
            vpos += ssss
            val = int(bits, 2)
            if val < (1 << (ssss - 1)):
                val = val - (1 << ssss) + 1
            ac_values.append(val)
        # ZRL (ssss == 0): no value bits, no entry in ac_values

    # Rebuild blocks
    ac_idx = 0
    sym_idx = 0
    blocks_dec = []
    for blk_i in range(n_blocks):
        # DC
        zig = np.zeros(64, dtype=np.int32)
        zig[0] = dc_coeffs[blk_i]
        # AC
        ac_syms = []
        while True:
            if sym_idx >= len(ac_rs_symbols):
                break
            run, ssss = ac_rs_symbols[sym_idx]
            sym_idx += 1
            if run == 0 and ssss == 0:
                break  # EOB
            if ssss == 0:
                # ZRL: 16 zeros, no associated value
                ac_syms.append((run, 0))
            else:
                ac_syms.append((run, ac_values[ac_idx]))
                ac_idx += 1
        ac_coeffs = decode_ac_runlength(ac_syms)
        zig[1:64] = ac_coeffs
        block = izigzag(zig)
        dq_block = dequantize(block, encoded.quant_table)
        spatial = idct_2d(dq_block)
        blocks_dec.append(spatial)

    blocks_dec = np.array(blocks_dec)
    img = image_from_blocks(blocks_dec, orig_h, orig_w)
    return img


# ============================================================
# RGB (color) image support
# ============================================================

def encode_image_rgb(img: np.ndarray, quality: int = 50) -> List[EncodedImage]:
    """Encode an RGB image by encoding each channel independently."""
    encoded = []
    for c in range(3):
        encoded.append(encode_image(img[:, :, c], quality))
    return encoded


def decode_image_rgb(encoded_list: List[EncodedImage]) -> np.ndarray:
    """Decode RGB channels back to color image."""
    channels = [decode_image(e) for e in encoded_list]
    return np.stack(channels, axis=-1)


# ============================================================
# Complexity measurement
# ============================================================

def measure_complexity(img: np.ndarray, quality: int = 50) -> dict:
    """
    Measure encoding/decoding time and memory usage.
    Returns dict with timing and size metrics.
    """
    import tracemalloc

    # Encode timing & memory
    tracemalloc.start()
    t0 = time.perf_counter()
    encoded = encode_image(img, quality)
    t_enc = time.perf_counter() - t0
    _, mem_enc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Decode timing & memory
    tracemalloc.start()
    t0 = time.perf_counter()
    decoded = decode_image(encoded)
    t_dec = time.perf_counter() - t0
    _, mem_dec_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "quality": quality,
        "encode_time_s": t_enc,
        "decode_time_s": t_dec,
        "encode_memory_peak_kb": mem_enc_peak / 1024,
        "decode_memory_peak_kb": mem_dec_peak / 1024,
        "compression_ratio": encoded.compression_ratio,
        "bpp": encoded.bpp,
        "original_bytes": encoded.original_size_bits // 8,
        "compressed_bytes": encoded.compressed_size_bits // 8,
    }
