"""Replay container for the browser viewer (format ``HVR1``).

A replay is one gzip stream: ``b"HVR1"``, a little-endian uint32 header length, a UTF-8
JSON header padded with spaces to a 4-byte boundary, then raw little-endian arrays. The
header's ``arrays`` table gives each array's dtype, shape, byte offset (from the end of the
header, 4-byte aligned) and codec. Codec ``"delta-shuffle"`` (int16 only) stores
differences along axis 0 (wrapping) with all low bytes before all high bytes, which makes
smooth fields compress well. The reader in ``web/viewer.js`` mirrors :func:`read_replay`.
"""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"HVR1"
_DTYPES = ("uint8", "int16", "int32", "float32")


def _encode(array: np.ndarray, codec: str) -> bytes:
    little = array.astype(array.dtype.newbyteorder("<"))
    if codec == "raw":
        return little.tobytes()
    if codec != "delta-shuffle" or array.dtype != np.int16:
        raise ValueError(f"unsupported codec {codec!r} for {array.dtype}")
    delta = little.copy()
    delta[1:] = little[1:] - little[:-1]  # int16 arithmetic wraps; decoding wraps back
    return np.frombuffer(delta.tobytes(), np.uint8).reshape(-1, 2).T.tobytes()


def write_replay(path: str | Path, meta: dict, arrays: dict[str, tuple[np.ndarray, str]]) -> Path:
    """Write ``arrays`` (name -> (array, codec)) plus JSON-serialisable ``meta``."""
    table, blobs, offset = {}, [], 0
    for name, (array, codec) in arrays.items():
        array = np.ascontiguousarray(array)
        if array.dtype.name not in _DTYPES:
            raise ValueError(f"{name}: unsupported dtype {array.dtype}")
        blob = _encode(array, codec)
        table[name] = {"dtype": array.dtype.name, "shape": list(array.shape),
                       "offset": offset, "codec": codec}
        blob += b"\0" * (-len(blob) % 4)
        blobs.append(blob)
        offset += len(blob)
    header = json.dumps({"meta": meta, "arrays": table}, allow_nan=False).encode()
    header += b" " * (-len(header) % 4)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = MAGIC + struct.pack("<I", len(header)) + header + b"".join(blobs)
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    return path


def read_replay(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    payload = gzip.decompress(Path(path).read_bytes())
    if payload[:4] != MAGIC:
        raise ValueError("not an HVR1 replay")
    size = struct.unpack("<I", payload[4:8])[0]
    header = json.loads(payload[8 : 8 + size])
    body = 8 + size
    arrays = {}
    for name, spec in header["arrays"].items():
        dtype = np.dtype(spec["dtype"]).newbyteorder("<")
        count = int(np.prod(spec["shape"]))
        raw = np.frombuffer(payload, np.uint8, count * dtype.itemsize, body + spec["offset"])
        if spec["codec"] == "delta-shuffle":
            interleaved = raw.reshape(2, -1).T.tobytes()
            delta = np.frombuffer(interleaved, dtype).reshape(spec["shape"])
            arrays[name] = np.cumsum(delta, axis=0, dtype=np.int16)
        else:
            arrays[name] = np.frombuffer(raw.tobytes(), dtype).reshape(spec["shape"])
    return header["meta"], arrays
