"""CRC for unitree_hg LowCmd, as checked by the real G1's motor controller.

Unitree's crc32_core() runs an MSB-first CRC-32 (poly 0x04C11DB7, init 0xFFFFFFFF, no reflection, no final
xor) over the packed LowCmd struct as 32-bit little-endian words, excluding the trailing crc word. That is
CRC-32/MPEG-2 over each word's big-endian bytes; this module uses a byte table for speed and keeps the
bitwise reference for the unit test. Packed layout (unitree_sdk2_python): '<2B2x' + 'B3x5fI'*35 + '5I'.
"""
import struct

import numpy as np

POLY = 0x04C11DB7
FMT = "<2B2x" + "B3x5fI" * 35 + "5I"
SIZE = struct.calcsize(FMT)  # 1004

_TABLE = []
for _b in range(256):
    _c = _b << 24
    for _ in range(8):
        _c = ((_c << 1) ^ POLY) & 0xFFFFFFFF if _c & 0x80000000 else (_c << 1) & 0xFFFFFFFF
    _TABLE.append(_c)
_TABLE = tuple(_TABLE)


def crc32_core_reference(words):
    """Bitwise port of unitree_sdk2 crc32_core (slow; for tests)."""
    crc = 0xFFFFFFFF
    for data in words:
        xbit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ POLY
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if data & xbit:
                crc ^= POLY
            xbit >>= 1
    return crc


def crc32_words(words_le_bytes):
    """Table CRC over little-endian 32-bit words given as bytes (length multiple of 4)."""
    be = np.frombuffer(words_le_bytes, dtype="<u4").astype(">u4").tobytes()
    crc = 0xFFFFFFFF
    t = _TABLE
    for byte in be:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ t[((crc >> 24) ^ byte) & 0xFF]
    return crc


def pack_lowcmd(msg):
    vals = [msg.mode_pr, msg.mode_machine]
    for m in msg.motor_cmd:
        vals += [m.mode, m.q, m.dq, m.tau, m.kp, m.kd, m.reserve]
    vals += list(msg.reserve) + [0]
    return struct.pack(FMT, *vals)


def lowcmd_crc(msg):
    return crc32_words(pack_lowcmd(msg)[:-4])
