"""Launch 4 Malmo clients and start training.

Usage:
    python run/start.py                        # Stage 1, with UI
    python run/start.py --headless             # No Minecraft windows
    python run/start.py --stage 1 --episodes 100
    python run/start.py --load                 # Resume from last checkpoint
"""

import subprocess
import time
import socket
import argparse
import sys
import os

MALMO_DIR = r"C:\Users\hiwa\Malmo_Python3.7\Minecraft"
PORTS = [10000, 10001, 10002, 10003]


def port_is_open(port: int) -> bool:
    """Check if a port is listening."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(('127.0.0.1', port))
        s.close()
        return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


def launch_clients(headless: bool = False):
    """Launch 4 Malmo Minecraft clients."""
    procs = []

    for port in PORTS:
        if port_is_open(port):
            print(f"  Port {port} already in use — skipping")
            continue

        print(f"  Launching client on port {port}...", end="", flush=True)

        # Build launch command
        cmd = f'cd /d "{MALMO_DIR}" && launchClient.bat --port {port}'
        if headless:
            # Headless: minimize window, lower priority
            proc = subprocess.Popen(
                f'cmd /c "{cmd}"',
                creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.SW_HIDE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            proc = subprocess.Popen(
                f'start "Malmo-{port}" cmd /c "{cmd}"',
                shell=True,
            )

        procs.append((port, proc))
        print(" launched")

        # Wait for THIS client to be ready before launching next one
        # Minecraft instances are heavy — launching too fast crashes them
        if port != PORTS[-1]:
            print(f"    Waiting for port {port} to be ready...", end="", flush=True)
            for _ in range(120):  # up to 2 minutes per client
                if port_is_open(port):
                    print(" ready!")
                    time.sleep(3)  # extra buffer
                    break
                time.sleep(1)
                print(".", end="", flush=True)
            else:
                print(f" timeout (will retry later)")

    return procs


def wait_for_clients(headless: bool = False, timeout: int = 300, max_retries: int = 3):
    """Wait until all 4 ports are listening. Retry failed ones."""
    print("\nWaiting for all clients to reach DORMANT state...")

    for attempt in range(max_retries):
        start = time.time()
        while time.time() - start < timeout:
            ready = [port_is_open(p) for p in PORTS]
            count = sum(ready)
            status = " ".join(f"{p}:{'OK' if r else '--'}" for p, r in zip(PORTS, ready))
            print(f"\r  [{count}/4] {status}", end="", flush=True)

            if all(ready):
                print(f"\n  All 4 clients ready! ({time.time() - start:.0f}s)")
                return True

            time.sleep(5)

        # Some clients didn't start — retry the failed ones
        failed = [p for p, r in zip(PORTS, ready) if not r]
        if not failed:
            return True

        if attempt < max_retries - 1:
            print(f"\n  Retrying {len(failed)} failed clients: {failed} "
                  f"(attempt {attempt + 2}/{max_retries})")
            for port in failed:
                print(f"    Relaunching port {port}...")
                cmd = f'cd /d "{MALMO_DIR}" && launchClient.bat --port {port}'
                if headless:
                    subprocess.Popen(
                        f'cmd /c "{cmd}"',
                        creationflags=subprocess.CREATE_NEW_CONSOLE | 0x00000080,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    subprocess.Popen(
                        f'start "Malmo-{port}" cmd /c "{cmd}"',
                        shell=True,
                    )
                # Wait for this one before next
                print(f"    Waiting for port {port}...", end="", flush=True)
                for _ in range(120):
                    if port_is_open(port):
                        print(" ready!")
                        time.sleep(3)
                        break
                    time.sleep(1)
                    print(".", end="", flush=True)
                else:
                    print(" still not ready")

    ready = [port_is_open(p) for p in PORTS]
    print(f"\n  {sum(ready)}/4 clients ready after {max_retries} attempts.")
    return all(ready)


def main():
    parser = argparse.ArgumentParser(description="Launch Malmo clients and train")
    parser.add_argument("--headless", action="store_true",
                        help="Run Minecraft clients without visible windows")
    parser.add_argument("--stage", type=int, default=1,
                        help="Training stage (default: 1)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of training episodes")
    parser.add_argument("--load", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--speed", type=int, default=None,
                        help="Game speed multiplier (default: use train script's GAME_SPEED)")
    parser.add_argument("--skip-launch", action="store_true",
                        help="Skip launching clients (they're already running)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  HALF STUPID — Stage {args.stage} Training")
    print(f"  {'Headless' if args.headless else 'With UI'} | {args.episodes} episodes")
    print("=" * 60)
    print()

    # Step 1: Launch clients
    if not args.skip_launch:
        print("Step 1: Launching 4 Minecraft clients...")
        launch_clients(headless=args.headless)
    else:
        print("Step 1: Skipping client launch (--skip-launch)")

    # Step 2: Wait for all clients
    print()
    if not wait_for_clients(headless=args.headless):
        print("\nFailed to start all clients. Exiting.")
        sys.exit(1)

    # Step 3: Small delay for stability
    print("\nWaiting 5s for clients to stabilize...")
    time.sleep(5)

    # Step 4: Start training
    print(f"\nStep 2: Starting Stage {args.stage} training...")
    print()

    train_cmd = [
        sys.executable,
        f"run/train_stage{args.stage}.py",
        "--episodes", str(args.episodes),
        "--ports", ",".join(str(p) for p in PORTS),
    ]
    if args.speed is not None:
        train_cmd.extend(["--speed", str(args.speed)])
    if args.load:
        # Find latest checkpoint
        save_dir = f"run/checkpoints/stage{args.stage}"
        if os.path.exists(save_dir):
            train_cmd.extend(["--load-dir", save_dir])

    try:
        proc = subprocess.run(train_cmd)
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
