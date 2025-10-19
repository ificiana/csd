import json
import mmap
import os


class MMapJSON:
    """
    A simple memory-mapped JSON file wrapper for fast telemetry.
    Automatically expands if data exceeds current mmap size.
    Only stores the latest JSON snapshot.
    """

    def __init__(self, path: str, size: int = 4096):
        """
        Args:
            path: Path to the mmap file.
            size: Max size in bytes for JSON content.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.size = size

        # Ensure the file exists and has the correct size
        if not os.path.exists(self.path):
            with open(self.path, "wb") as f:
                f.write(b"\x00" * self.size)

        # Open the file for read/write memory mapping
        self.fp = open(self.path, "r+b")
        self.mmap = mmap.mmap(self.fp.fileno(), self.size)

    def _resize(self, new_size: int):
        """Resize the mmap file safely."""
        self.mmap.close()
        self.fp.close()

        # Expand the file
        with open(self.path, "r+b") as f:
            f.seek(new_size - 1)
            f.write(b"\x00")

        # Reopen and remap
        self.size = new_size
        self.fp = open(self.path, "r+b")
        self.mmap = mmap.mmap(self.fp.fileno(), self.size)
        print(f"[WARN] [MMapJSON] mmap size increased to {self.size} bytes")

    def write(self, data: dict | list):
        """Write a JSON snapshot to the mmap file."""
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
        # Clear old content
        self.mmap.seek(0)
        self.mmap.write(b"\x00" * self.size)
        self.mmap.seek(0)

    def read(self) -> dict | None:
        """Read the latest JSON snapshot from the mmap file."""
        self.mmap.seek(0)
        raw = self.mmap.read(self.size).rstrip(b"\x00")
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def close(self):
        self.mmap.close()
        self.fp.close()
