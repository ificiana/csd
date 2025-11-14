import json
import mmap
import os

from config import MMAP_SIZE


class MMapJSON:
    """
    A unified memory-mapped JSON wrapper.
    Supports both file-backed and shared-memory modes.
    Automatically expands if data exceeds current mmap size.
    Only stores the latest JSON snapshot.
    """

    def __init__(self, path_or_topic: str, size: int = MMAP_SIZE, file: bool = False):
        """
        Args:
            path_or_topic: Path to mmap file (if file=True) or shared memory topic name (if file=False).
            size: Initial size in bytes for JSON content.
            file: If True, uses a file-backed mmap; if False, uses shared memory (Windows-only tagname).
        """
        self.file_mode = file
        self.size = size

        if self.file_mode:
            # File-backed mmap
            self.path = path_or_topic
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

            if not os.path.exists(self.path):
                with open(self.path, "wb") as f:
                    f.write(b"\x00" * self.size)

            self.fp = open(self.path, "r+b")
            self.mmap = mmap.mmap(self.fp.fileno(), self.size)
        else:
            # Shared memory mmap (e.g. for inter-process communication)
            self.topic = path_or_topic
            self.fp = None
            self.mmap = mmap.mmap(-1, self.size, tagname=self.topic)

    def _resize(self, new_size: int):
        """Resize the mmap safely."""
        self.mmap.close()
        if self.fp:
            self.fp.close()

        if self.file_mode:
            # Expand file
            with open(self.path, "r+b") as f:
                f.seek(new_size - 1)
                f.write(b"\x00")

            # Remap
            self.fp = open(self.path, "r+b")
            self.mmap = mmap.mmap(self.fp.fileno(), new_size)
        else:
            # Remap shared memory
            self.mmap = mmap.mmap(-1, new_size, tagname=self.topic)

        self.size = new_size
        print(f"[WARN] [MMapJSON] mmap size increased to {self.size} bytes")

    def write(self, data: dict | list):
        """Write a JSON snapshot to the mmap."""
        json_bytes = json.dumps(data).encode("utf-8")

        # Auto-expand if too large
        if len(json_bytes) > self.size:
            new_size = self.size
            while len(json_bytes) > new_size:
                new_size *= 2
            self._resize(new_size)

        # Clear old content
        self.clear()
        self.mmap.write(json_bytes)
        self.mmap.flush()

    def clear(self):
        """Clear the mmap contents."""
        self.mmap.seek(0)
        self.mmap.write(b"\x00" * self.size)
        self.mmap.seek(0)

    def read(self) -> dict | None:
        """Read the latest JSON snapshot from the mmap."""
        self.mmap.seek(0)
        raw = self.mmap.read(self.size).rstrip(b"\x00")
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def close(self):
        """Close mmap and file handle if applicable."""
        self.mmap.close()
        if self.fp:
            self.fp.close()
