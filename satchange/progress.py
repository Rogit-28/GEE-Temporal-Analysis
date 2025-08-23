"""
Progress indicators for SatChange CLI operations.

Provides Rich-based progress bars and spinners that gracefully degrade
if Rich is not installed.
"""

import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
        TimeElapsedColumn,
    )
    from rich.console import Console

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


@contextmanager
def spinner(description: str):
    """Context manager for indeterminate operations (spinners).

    Usage:
        with spinner("Authenticating with GEE..."):
            gee_client.authenticate()
    """
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(description, total=None)
            yield progress
    else:
        # Fallback: just print start/end
        import click

        click.echo(f"  {description}")
        yield None


@contextmanager
def progress_bar(description: str, total: int):
    """Context manager for determinate operations (progress bars).

    Usage:
        with progress_bar("Downloading images", total=6) as update:
            for band in bands:
                download(band)
                update(1)
    """
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task(description, total=total)

            def advance(step: int = 1):
                progress.advance(task_id, step)

            yield advance
    else:
        # Fallback: print progress manually
        import click

        completed = [0]

        def advance(step: int = 1):
            completed[0] += step
            click.echo(f"  {description}: {completed[0]}/{total}")

        yield advance


def print_status(message: str, style: str = "bold green"):
    """Print a styled status message."""
    if RICH_AVAILABLE:
        console.print(f"[{style}]{message}[/{style}]")
    else:
        import click

        click.echo(message)


def print_error(message: str):
    """Print a styled error message."""
    if RICH_AVAILABLE:
        console.print(f"[bold red]{message}[/bold red]")
    else:
        import click

        click.echo(message, err=True)
