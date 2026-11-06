import json
import os
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def read(path):
    return json.loads(Path(path).read_text())
