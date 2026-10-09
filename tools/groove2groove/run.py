"""Run the unmodified upstream pretrained model in a separate process."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "inference_workspace/groove2groove"


def run_pair(content, style, output, workspace=WORKSPACE, seed=42, sample=True):
    output = Path(output).resolve()
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(workspace / "env/bin/python"), "-m",
        "groove2groove.models.roll2seq_style_transfer",
        "--logdir", str(workspace / "checkpoints/v01"), "run-midi",
        "--batch-size", "1", "--bars-per-segment", "8", "--seed", str(seed),
        "--softmax-temperature", "0.6",
    ]
    if sample:
        command.append("--sample")
    command.extend([str(Path(content).resolve()), str(Path(style).resolve()), str(output)])
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
                        "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "2",
                        "TF_CPP_MIN_LOG_LEVEL": "2"})
    started = time.monotonic()
    with output.with_suffix(".log").open("x") as log:
        result = subprocess.run(command, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, timeout=600)
    record = {"content": str(content), "style": str(style), "seed": seed,
              "sample": sample, "temperature": 0.6, "command": command,
              "seconds": round(time.monotonic() - started, 2),
              "returncode": result.returncode}
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError("Inference failed; see " + str(output.with_suffix(".log")))
    print("Generated {} in {} s".format(output, record["seconds"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("content", type=Path)
    parser.add_argument("style", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    run_pair(args.content, args.style, args.output, args.workspace.resolve(),
             args.seed, not args.greedy)


if __name__ == "__main__":
    main()
