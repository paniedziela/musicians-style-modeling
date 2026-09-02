from __future__ import annotations
import argparse
from .experiment import audit_e4, load_e4_config
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m musicians_style.e4")
    parser.add_argument("--config", required=True); parser.add_argument("--stage", required=True, choices=("audit",))
    args = parser.parse_args(argv); print(audit_e4(load_e4_config(args.config))); return 0
if __name__ == "__main__": raise SystemExit(main())
