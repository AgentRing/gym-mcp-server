"""Pytest configuration and fixtures."""

import warnings

# Suppress deprecation warnings from third-party dependencies
warnings.filterwarnings(
    "ignore",
    message=".*websockets.*deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*pkg_resources.*deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Deprecated call to.*pkg_resources.*",
    category=DeprecationWarning,
)

