import os
import platform
import subprocess
import time
from pathlib import Path

import requests


class LumaServer:
    def __init__(self, port=1234):
        self.port = port
        self.process = None
        self.binary_path = self._get_binary_path()

    def _get_binary_path(self):
        """Determines the correct binary for the current OS."""
        system = platform.system().lower()

        base_path = Path(__file__).parent / "bin"

        if system == "linux":
            filename = "luma-linux-amd64"
        elif system == "windows":
            filename = "luma-windows-amd64.exe"
        elif system == "darwin":  # MacOS
            filename = "luma-macos-amd64"
        else:
            raise OSError(f"Unsupported operating system: {system}")

        binary = base_path / filename

        if not binary.exists():
            raise FileNotFoundError(f"Luma binary not found at: {binary}. Please reinstall the package.")

        # Ensure execution permissions on Unix
        if system != "windows":
            st = os.stat(binary)
            os.chmod(binary, st.st_mode | 0o111)

        return str(binary)

    def start(self):
        """Starts the Luma server in a subprocess."""
        if self.is_running():
            print(f"Luma is already running on port {self.port}")
            return

        print(f"Starting Luma server from {self.binary_path}...")

        # Run process detached/background
        try:
            self.process = subprocess.Popen(
                [self.binary_path],
                stdout=subprocess.DEVNULL,  # Redirect logs if needed
                stderr=subprocess.DEVNULL,
            )

            # Wait for health check
            self._wait_for_startup()
            print(f"Luma Server started successfully on port {self.port}")
        except Exception as e:
            raise RuntimeError(f"Failed to start Luma server: {e}")

    def stop(self):
        """Terminates the server process."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            print("Luma Server stopped.")

    def is_running(self):
        """Checks if the server API is responsive."""
        try:
            response = requests.get(f"http://localhost:{self.port}/health", timeout=1)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _wait_for_startup(self, timeout=5):
        start = time.time()
        while time.time() - start < timeout:
            if self.is_running():
                return
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for Luma server to start.")


# Singleton instance for easy access
default_server = LumaServer()
