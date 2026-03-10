"""Generate Python gRPC stubs from the substructure worker.proto."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PROTO_DIR = ROOT / ".." / "substructure" / "proto"
OUT_DIR = ROOT / "substructure" / "_proto"


def main() -> None:
    proto_file = PROTO_DIR / "worker.proto"
    if not proto_file.exists():
        print(f"error: {proto_file} not found", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        f"--pyi_out={OUT_DIR}",
        str(proto_file),
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd)

    # grpc-tools generates absolute imports (e.g. `import worker_pb2`).
    # Rewrite them to relative imports so they work inside the package.
    for py_file in OUT_DIR.glob("*.py"):
        text = py_file.read_text()
        fixed = re.sub(
            r"^import (\w+_pb2)",
            r"from . import \1",
            text,
            flags=re.MULTILINE,
        )
        if fixed != text:
            py_file.write_text(fixed)
            print(f"  fixed imports in {py_file.name}")

    print(f"stubs generated in {OUT_DIR}")


if __name__ == "__main__":
    main()
