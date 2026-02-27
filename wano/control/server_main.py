import argparse
import sys
import traceback
from pathlib import Path

try:
    from wano.control.server import init_control_plane, run_server
except (ImportError, ModuleNotFoundError) as e:
    print(f"Error importing server module: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Wano control plane server")
    parser.add_argument("port", type=int, help="API server port")
    parser.add_argument("ray_port", type=int, help="Ray head port")
    parser.add_argument("db_path", type=Path, help="SQLite database path")
    args = parser.parse_args()
    try:
        args.db_path.parent.mkdir(parents=True, exist_ok=True)
        init_control_plane(args.db_path, args.ray_port, args.port)
        run_server(port=args.port)
    except (ValueError, OSError, RuntimeError) as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
