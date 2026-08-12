"""Top-level package entry point."""

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="face-destyle",
        description=(
            "Face destylization research utilities. Use scripts/ for stage-specific commands."
        ),
    )
    parser.add_argument("--version", action="store_true", help="Print the package version.")
    args = parser.parse_args(argv)
    if args.version:
        from face_destyle import __version__

        print(__version__)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
