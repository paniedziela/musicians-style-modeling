"""Run a small 3-by-3 comparison, including self-reference controls."""

import argparse
from pathlib import Path

from prepare_examples import PIECES, excerpt
from run import run_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asap", type=Path, default=Path("datasets/asap-dataset-1.2"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = args.output / "inputs"
    inputs.mkdir()
    for name, path in PIECES.items():
        excerpt(args.asap / path, inputs / (name + ".mid"))
    for content in PIECES:
        for style in PIECES:
            output = args.output / "outputs" / (content + "_to_" + style + "_seed42.mid")
            run_pair(inputs / (content + ".mid"), inputs / (style + ".mid"), output)


if __name__ == "__main__":
    main()
