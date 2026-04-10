"""CLI entry point: python -m torch_recall <command> [args]"""

import argparse
import json
import sys


def cmd_encode_user(args: argparse.Namespace) -> None:
    from torch_recall.recall_method.targeting.encoder import save_user_tensors

    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    user_attrs = json.loads(args.user)
    save_user_tensors(user_attrs, meta, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m torch_recall")
    sub = parser.add_subparsers(dest="command")

    enc_user = sub.add_parser("encode-user", help="Encode user attributes for targeting recall")
    enc_user.add_argument("--user", required=True, help='User attributes as JSON, e.g. \'{"city":"北京","age":25}\'')
    enc_user.add_argument("--meta", required=True, help="Path to targeting_meta.json")
    enc_user.add_argument("--output", required=True, help="Output path for tensors.pt")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "encode-user":
        cmd_encode_user(args)


if __name__ == "__main__":
    main()
