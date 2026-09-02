from __future__ import annotations

import argparse

from .experiment import audit_e4, evaluate_e4, load_e4_config, prepare_e4, report_e4, smoke_e4, train_e4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m musicians_style.e4")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("audit", "prepare", "smoke", "train", "evaluate", "report"))
    args = parser.parse_args(argv)
    actions = {"audit": audit_e4, "prepare": prepare_e4, "smoke": smoke_e4, "train": train_e4, "evaluate": evaluate_e4, "report": report_e4}
    print(actions[args.stage](load_e4_config(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
