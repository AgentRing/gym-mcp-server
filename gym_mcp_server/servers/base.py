"""
Abstract base class for gym environment servers.

This module defines the common interface that all server implementations
(MCP, HTTP, gRPC) must follow.
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional

from ..service import GymService

logger = logging.getLogger(__name__)


class BaseGymServer(ABC):
    """Abstract base class for gym environment servers.

    This class defines the common interface and shared functionality
    for all server implementations (MCP, HTTP, gRPC).

    All servers share:
    - Environment ID and render mode
    - Shared service layer (GymService)
    - Non-blocking mode support (start/stop/is_alive)
    - Threading support for background execution
    """

    def __init__(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
    ):
        """Initialize the base gym server.

        Args:
            env_id: The Gymnasium environment ID
            render_mode: Optional render mode for the environment
        """
        self.env_id = env_id
        self.render_mode = render_mode

        # Initialize the shared service layer
        self._service = GymService(env_id=env_id, render_mode=render_mode)

        # Threading support for non-blocking mode
        self._thread: Optional[threading.Thread] = None

    @property
    def service(self) -> GymService:
        """Get the underlying gym service instance.

        Returns:
            The GymService instance
        """
        return self._service

    @abstractmethod
    def run(self, **kwargs) -> None:
        """Run the server (blocking).

        This method should start the server and block until it's stopped.
        The exact signature depends on the server type.

        Args:
            **kwargs: Server-specific arguments (host, port, transport, etc.)
        """
        pass

    def start(self, **kwargs) -> None:
        """Start the server in a background thread (non-blocking).

        This allows running the server while continuing to execute other code
        in the main thread. Useful for testing, embedded applications, or when
        you need to run server and client in the same Python process.

        Args:
            **kwargs: Server-specific arguments passed to run()

        Raises:
            RuntimeError: If server is already running
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Server already started")

        self._thread = threading.Thread(
            target=self._run_in_thread, args=(), kwargs=kwargs, daemon=True
        )
        self._thread.start()

        # Wait a bit for the server to start up
        import time

        logger.info("Waiting for server to start...")
        time.sleep(1)

    def _run_in_thread(self, **kwargs) -> None:
        """Internal method to run server in a thread.

        Args:
            **kwargs: Arguments passed to run()
        """
        try:
            self.run(**kwargs)
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)

    @abstractmethod
    def stop(self, **kwargs) -> None:
        """Stop the server if running in non-blocking mode.

        Args:
            **kwargs: Server-specific stop arguments (grace period, etc.)
        """
        pass

    def is_alive(self) -> bool:
        """Check if the server thread is alive.

        Returns:
            True if the server thread is running, False otherwise
        """
        return self._thread is not None and self._thread.is_alive()

