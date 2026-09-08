#!/usr/bin/env python3
"""Link the host observer against exactly the native compiler's built libraries."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    args = parser.parse_args()
    build = args.build.resolve()
    source = Path(__file__).with_name("plan_memory_oracle.cpp")
    commands = json.loads((build / "compile_commands.json").read_text())
    entry, = [c for c in commands if c["file"].endswith("/tools/bishengir-compile/bishengir-compile.cpp")]
    command = shlex.split(entry["command"])
    native_object = command[command.index("-o") + 1]
    oracle_object = str(build / "plan-memory-oracle.o")
    command[command.index("-o") + 1] = oracle_object
    command[command.index("-c") + 1] = str(source)
    subprocess.run(command, cwd=build, check=True)
    cache = (build / "CMakeCache.txt").read_text().splitlines()
    ninja, = [line.split("=", 1)[1] for line in cache if line.startswith("CMAKE_MAKE_PROGRAM:")]
    lines = subprocess.check_output([ninja, "-t", "commands", "bin/bishengir-compile"], cwd=build, text=True).splitlines()
    link = shlex.split(lines[-1])
    if link[:2] == [":", "&&"]: link = link[2:]
    if link[-2:] == ["&&", ":"]: link = link[:-2]
    if "&&" in link: raise RuntimeError("Unexpected compound native link command")
    link[link.index(native_object)] = oracle_object
    link[link.index("-o") + 1] = "bin/plan-memory-oracle"
    subprocess.run(link, cwd=build, check=True)
    (build / "oracle-build-commands.json").write_text(json.dumps([command, link], indent=2) + "\n")


if __name__ == "__main__":
    main()
