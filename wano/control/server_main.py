import sys
import traceback
from pathlib import Path

try:
    from wano.control.server import init_control_plane, run_server
except Exception as e:
    print(f"Error importing server module: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)


def main():
    if len(sys.argv) < 4:
        print(
            f"Usage: server_main.py <port> <ray_port> <db_path>\nGot {len(sys.argv)} arguments: {sys.argv}",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        port, ray_port, db_path = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        init_control_plane(db_path, ray_port, port)
        run_server(port=port)
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
