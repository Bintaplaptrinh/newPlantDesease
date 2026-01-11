from __future__ import annotations

import argparse

from src.pipeline import ensure_web_placeholders, prepare_and_export_web_data


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(prog="newplant", description="NewPlantDesease pipeline (no training executed by default)")
	sub = p.add_subparsers(dest="cmd", required=True)

	prep = sub.add_parser("prepare-data", help="Download+merge dataset and export dataset JSON for the web")
	prep.add_argument("--image-size", type=int, default=224)
	prep.add_argument("--batch-size", type=int, default=64)
	prep.add_argument("--num-workers", type=int, default=8)
	prep.add_argument("--data-root", type=str, default="data")

	sub.add_parser("placeholders", help="Create empty JSON placeholders for training/eval artifacts")

	serve = sub.add_parser("serve", help="Run Flask API server")
	serve.add_argument("--host", type=str, default="0.0.0.0")
	serve.add_argument("--port", type=int, default=5000)
	serve.add_argument("--debug", action="store_true")

	return p


def main(argv: list[str] | None = None) -> int:
	parser = _build_parser()
	args = parser.parse_args(argv)

	if args.cmd == "prepare-data":
		prepare_and_export_web_data(
			image_size=args.image_size,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			data_root=args.data_root,
		)
		return 0

	if args.cmd == "placeholders":
		ensure_web_placeholders()
		return 0

	if args.cmd == "serve":
		from src.server.app import create_app

		app = create_app()
		app.run(host=args.host, port=args.port, debug=bool(args.debug))
		return 0

	raise RuntimeError("unreachable")


if __name__ == "__main__":
	raise SystemExit(main())
