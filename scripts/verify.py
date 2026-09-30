from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path = ROOT) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    run([sys.executable, "-m", "compileall", "-q", "backend/src", "backend/tests", "datasets/seed.py"])
    run([sys.executable, "-m", "pytest", "-q", "backend/tests"])

    npm = shutil.which("npm")
    node_modules = ROOT / "frontend" / "node_modules"
    if npm and node_modules.exists():
        run([npm, "run", "build"], cwd=ROOT / "frontend")
    else:
        print("Frontend: se omite ng build porque node_modules no está instalado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
