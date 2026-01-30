import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


class LumaServer:
    def __init__(self, port=1234, host="0.0.0.0", data_dir=None):
        self.port = port
        self.host = host
        self.process = None
        self.binary_path = self._get_binary_path()

        if data_dir:
            self.data_dir = Path(data_dir).resolve()
        else:
            self.data_dir = Path.home() / "data_luma"

    def _get_binary_path(self):
        """Determines the correct binary path for the current OS."""
        system = platform.system().lower()
        base_path = Path(__file__).parent / "bin"

        if system == "linux":
            filename = "luma-linux-amd64"
        elif system == "windows":
            filename = "luma-windows-amd64.exe"
        elif system == "darwin":
            filename = "luma-macos-amd64"
        else:
            logger.error(f"Operating system '{system}' is not supported.")
            raise OSError(f"Unsupported OS: {system}")

        binary = base_path / filename

        if not binary.exists():
            logger.critical(f"Luma binary not found at: {binary}")
            raise FileNotFoundError(f"Binary not found at: {binary}")

        if system != "windows":
            try:
                st = os.stat(binary)
                os.chmod(binary, st.st_mode | 0o111)
            except Exception as e:
                logger.warning(f"Could not set execution permissions: {e}")

        return str(binary)

    def start(self):
        """Starts the Luma server."""
        if self.is_running():
            logger.warning(f"Luma server is already running on port {self.port}")
            return

        # 1. Ensure data directory
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting Luma server...")

        cmd = [
            self.binary_path,
            "--port",
            str(self.port),
            "--bind",
            self.host,
            "--DATA_DIR",
            str(self.data_dir),
        ]

        try:
            # Redirect stdout/stderr so logs appear in console
            self.process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

            self._wait_for_startup()

            # --- NOTIFICACIÓN DE SEGURIDAD ---
            # Si no se ha inyectado una clave maestra específica en el env, asumimos modo DEV
            if "LUMA_API_KEY" not in os.environ:
                url = f"http://{self.host}:{self.port}"
                logger.warning("┌──────────────────────────────────────────────────────────────┐")
                logger.warning("│ ⚠️  SECURITY NOTICE: Running with default API Key 'dev'      │")
                logger.warning("│                                                              │")
                logger.warning(f"│ 1. Access Dashboard: {url:<36}│")
                logger.warning("│ 2. Login with Key:   'dev'                                   │")
                logger.warning("│ 3. Action Required:  Create a new API Key & delete 'dev'     │")
                logger.warning("└──────────────────────────────────────────────────────────────┘")

            logger.info(f"✅ Luma Server active on port {self.port}")

        except Exception as e:
            logger.error(f"Failed to start Luma server: {e}")
            self.stop()
            raise

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def is_running(self):
        """Health check tolerante a Auth (401)."""
        try:
            url = f"http://localhost:{self.port}/health"
            response = requests.get(url, timeout=0.5)
            # 200 = OK, 401 = OK (pero requiere auth), 403 = OK (Forbidden)
            return response.status_code in [200, 401, 403]
        except requests.RequestException:
            return False

    def _wait_for_startup(self, timeout=5):
        if self.process is None:
            raise RuntimeError("Intentando esperar a un proceso que no ha sido iniciado (self.process es None).")

        start = time.time()
        while time.time() - start < timeout:
            if self.process.poll() is not None:
                raise RuntimeError(f"Luma process exited with code {self.process.returncode}")

            if self.is_running():
                return

            time.sleep(0.1)

        raise TimeoutError(f"Timed out waiting for Luma on port {self.port}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
