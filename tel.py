import json
import mmap
import os


class MMapJSON:
    """
    A simple memory-mapped JSON file wrapper for fast telemetry.
    Only stores the latest JSON snapshot.
    Ground station can poll the file safely.
    """

    def __init__(self, path: str, size: int = 65536):
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

    def write(self, data: dict):
        """Write a JSON snapshot to the mmap file."""
        json_bytes = json.dumps(data).encode("utf-8")
        if len(json_bytes) > self.size:
            raise ValueError("JSON data exceeds mmap size")

        # Clear old content
        self.mmap.seek(0)
        self.mmap.write(b"\x00" * self.size)
        self.mmap.seek(0)
        self.mmap.write(json_bytes)
        self.mmap.flush()

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
