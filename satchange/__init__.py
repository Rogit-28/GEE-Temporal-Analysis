"""
SatChange - CLI tool for detecting temporal changes in satellite imagery.

A Python-based CLI tool that enables users to detect and visualize temporal
changes in satellite imagery for specified geographic areas.
"""

__version__ = "1.0.0a1"
__author__ = "SatChange Team"
__email__ = "team@satchange.dev"

from .cli import main

__all__ = ["main"]
