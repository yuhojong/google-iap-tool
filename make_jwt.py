"""Utility script to generate an App Store Connect JWT."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from apple_store import AppleStoreConfigError, generate_jwt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a JWT for the App Store Connect API using environment configuration."
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help=(
            "Ignore any cached token and force generation of a new JWT. "
            "This flag has no effect if no token has been generated yet."
        ),
    )
    args = parser.parse_args()

    load_dotenv()

    try:
        token = generate_jwt(force_refresh=args.force_refresh)
    except AppleStoreConfigError as exc:
        print(f"환경 구성이 올바르지 않습니다: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - runtime safeguard
        print(f"JWT 생성 중 예기치 못한 오류가 발생했습니다: {exc}", file=sys.stderr)
        return 1

    print(token)
    return 0


if __name__ == "__main__":  # pragma: no mutate - CLI entry point
    raise SystemExit(main())
