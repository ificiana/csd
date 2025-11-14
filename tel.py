"""
Memory-mapped JSON wrapper for inter-process telemetry.

Provides efficient shared-memory communication between simulation process
and visualization/logging processes using memory-mapped files or Windows
shared memory regions.
"""

from io import BufferedRandom
import json
import mmap
import os

from config import MMAP_SIZE


class MMapJSON:
    """
    Memory-mapped JSON storage supporting file-backed and shared-memory modes.
    Automatically expands if data exceeds current size. Stores only latest snapshot.
    """

    def __init__(self, path_or_topic: str, size: int = MMAP_SIZE, file: bool = False):
        """
        Args:
            path_or_topic: File path (file=True) or shared memory topic (file=False).
            size: Initial buffer size in bytes.
            file: If True uses file-backed mmap, else Windows shared memory.
        """
        self.file_mode = file
        self.size = size

        if self.file_mode:
            self.path = path_or_topic
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

            if not os.path.exists(self.path):
                with open(self.path, "wb") as f:
                    f.write(b"\x00" * self.size)

            self.fp = open(self.path, "r+b")
            self.mmap = mmap.mmap(self.fp.fileno(), self.size)
        else:
            self.topic = path_or_topic
            self.fp = None
            self.mmap = mmap.mmap(-1, self.size, tagname=self.topic)

    def _resize(self, new_size: int):
        """Resizes mmap buffer to accommodate larger data."""
        self.mmap.close()
        if self.fp:
            self.fp.close()

        if self.file_mode:
            with open(self.path, "r+b") as f:
                f.seek(new_size - 1)
                f.write(b"\x00")

            self.fp = open(self.path, "r+b")
            self.mmap = mmap.mmap(self.fp.fileno(), new_size)
        else:
            self.mmap = mmap.mmap(-1, new_size, tagname=self.topic)

        self.size = new_size
        print(f"[WARN] [MMapJSON] mmap size increased to {self.size} bytes")

    def write(self, data: dict | list):
        """Writes JSON snapshot to mmap, expanding buffer if needed."""
        json_bytes = json.dumps(data).encode("utf-8")

        if len(json_bytes) > self.size:
            new_size = self.size
            while len(json_bytes) > new_size:
                new_size *= 2
            self._resize(new_size)

        self.clear()
        self.mmap.write(json_bytes)
        self.mmap.flush()

    def clear(self):
        """Clears mmap contents by writing zeros."""
        self.mmap.seek(0)
        self.mmap.write(b"\x00" * self.size)
        self.mmap.seek(0)

    def read(self) -> dict | None:
        """Reads latest JSON snapshot from mmap."""
        self.mmap.seek(0)
        raw = self.mmap.read(self.size).rstrip(b"\x00")
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def close(self):
        """Closes mmap and file handle if applicable."""
        self.mmap.close()
        if self.fp:
            self.fp.close()
