"""
CLI interface for SatChange - Command-line interface for satellite change detection.

This module provides the main entry point and command definitions for the SatChange CLI tool.
"""

import click
import json
import numpy as np
from typing import Optional
import os
import sys
from datetime import datetime

from .config import Config
from .gee_client import (
    GEEClient,
    QuotaExceededError,
    RateLimitError,
    NoImageryError,
    DownloadError,
)
from .cache import CacheManager
from .image_processor import ImageProcessor
from .change_detector import ChangeDetector
from .visualization import VisualizationManager
from .utils import setup_logging, parse_date, parse_coordinates, NumpyJSONEncoder
from .progress import spinner, progress_bar, print_status, print_error


def format_location_name(lat: float, lon: float) -> str:
    """Format lat/lon as a location name with up to 4 decimal places."""
    lat_str = f"{lat:.4f}".rstrip("0").rstrip(".")
    lon_str = f"{lon:.4f}".rstrip("0").rstrip(".")
    return f"{lat_str}_{lon_str}"


def generate_output_prefix(
    name: Optional[str], lat: float, lon: float, start_date: str, end_date: str
) -> str:
    """Generate output filename prefix following PRD naming convention.

    Format: {location_name}_{start_date}_{end_date}
    """
    location_name = name if name else format_location_name(lat, lon)
    # Ensure dates are in YYYY-MM-DD format (remove any time component)
    start = start_date.split("T")[0] if "T" in start_date else start_date
    end = end_date.split("T")[0] if "T" in end_date else end_date
    return f"{location_name}_{start}_{end}"


@click.group()
@click.version_option()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--config-file", type=click.Path(), help="Path to configuration file")
@click.pass_context
def main(ctx: click.Context, verbose: bool, config_file: Optional[str]) -> None:
    """
    SatChange - Detect temporal changes in satellite imagery.

    A CLI tool for analyzing how geographic areas have changed over time using
    satellite imagery from Google Earth Engine.
    """
    ctx.ensure_object(dict)

    # Setup logging
    setup_logging(verbose)

    # Load configuration
    config = Config(config_file)
    ctx.obj["config"] = config

    # Initialize GEE client if authenticated
    if config.is_authenticated():
        ctx.obj["gee_client"] = GEEClient(config)
    else:
        ctx.obj["gee_client"] = None


@main.group()
def config() -> None:
    """Manage SatChange configuration."""
    pass


@config.command()
@click.option(
    "--service-account-key",
    type=click.Path(exists=True),
    help="Path to GEE service account JSON key file",
)
@click.option("--project-id", help="Google Cloud project ID")
@click.pass_context
def init(
    ctx: click.Context, service_account_key: Optional[str], project_id: Optional[str]
) -> None:
    """
    Initialize SatChange configuration and authenticate with Google Earth Engine.
    """
    config = ctx.obj["config"]

    try:
        if service_account_key and project_id:
            config.initialize_auth(service_account_key, project_id)
            click.echo("[OK] Configuration initialized successfully")
            click.echo(f"  Service account: {config.get('service_account_email')}")
            click.echo(f"  Project ID: {config.get('project_id')}")
        else:
            click.echo("Please provide both --service-account-key and --project-id")
            click.echo(
                "Example: satchange config init --service-account-key /path/to/key.json --project-id my-project"
            )
            sys.exit(1)

    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"[ERROR] Configuration initialization failed: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--center", required=True, type=str, help="Lat,Lon (e.g., 13.0827,80.2707)"
)
@click.option("--size", default=100, type=int, help="Pixel dimensions (NxN)")
@click.option(
    "--date-range", required=True, type=str, help="Start:End (YYYY-MM-DD:YYYY-MM-DD)"
)
@click.option("--cloud-threshold", default=20, type=int, help="Max cloud coverage %")
@click.pass_context
def inspect(
    ctx: click.Context, center: str, size: int, date_range: str, cloud_threshold: int
) -> None:
    """
    Preview available Sentinel-2 scenes for a given area and date range.
    """
    gee_client = ctx.obj["gee_client"]
    if not gee_client:
        click.echo("[ERROR] Not authenticated with Google Earth Engine", err=True)
        click.echo("Run 'satchange config init' first", err=True)
        sys.exit(1)

    try:
        # Parse inputs
        lat, lon = parse_coordinates(center)
        start_date, end_date = parse_date(date_range)

        # Create bounding box
        bbox = gee_client.create_bbox(lat, lon, size)

        # Query GEE
        collection = gee_client.query_imagery(
            bbox, start_date, end_date, cloud_threshold
        )

        # Retrieve metadata
        scenes = gee_client.get_scenes_metadata(collection)

        # Display results
        click.echo(f"Found {len(scenes)} clear scenes")
        click.echo(f"\nTop 5 clearest:")
        for scene in sorted(scenes, key=lambda x: x["cloud_coverage"])[:5]:
            date = scene["date"]
            cloud_pct = scene["cloud_coverage"]
            click.echo(f"  {date} - {cloud_pct:.1f}% clouds")

    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"[ERROR] Inspection failed: {e}", err=True)
        sys.exit(1)


def display_cloud_check_result(result: dict, label: str) -> None:
    """Display cloud check result with formatting."""
    if not result["found"]:
        click.echo(f"  {label}: No image found for {result['date']}")
        return

    status = "[OK]" if result["is_good"] else "[WARN]"
    cloud_info = f"local={result['local_cloud_pct']:.1f}%, scene={result['scene_cloud_pct']:.1f}%"
    click.echo(f"  {label}: {result['date']} - {cloud_info} {status}")


def display_alternatives(alternatives: list, max_display: int = 5) -> None:
    """Display alternative dates with cloud coverage."""
    click.echo("\n  Available alternatives:")
    for i, alt in enumerate(alternatives[:max_display]):
        marker = " [RECOMMENDED]" if alt["is_recommended"] else ""
        click.echo(
            f"    [{i + 1}] {alt['date']} - local={alt['local_cloud_pct']:.1f}%{marker}"
        )

    if len(alternatives) > max_display:
        click.echo(f"    ... and {len(alternatives) - max_display} more")


def prompt_date_selection(alternatives: list, original_date: str, label: str) -> str:
    """Prompt user to select a date from alternatives or proceed with original."""
    click.echo(f"\n  Options for {label}:")
    click.echo(f"    [0] Keep original: {original_date} (not recommended - high cloud)")

    for i, alt in enumerate(alternatives[:5]):
        marker = " *" if alt["is_recommended"] else ""
        click.echo(
            f"    [{i + 1}] {alt['date']} - {alt['local_cloud_pct']:.1f}% cloud{marker}"
        )

    while True:
        choice = click.prompt("  Select option", type=int, default=1)
        if choice == 0:
            click.echo(f"  Proceeding with original date: {original_date}")
            return original_date
        elif 1 <= choice <= min(5, len(alternatives)):
            selected = alternatives[choice - 1]["date"]
            click.echo(f"  Selected: {selected}")
            return selected
        else:
            click.echo("  Invalid choice, try again")


@main.command()
@click.option(
    "--center", required=True, type=str, help="Lat,Lon (e.g., 13.0827,80.2707)"
)
@click.option("--size", default=100, type=int, help="Pixel dimensions (NxN)")
@click.option(
    "--date-a", required=True, type=str, help="First comparison date YYYY-MM-DD"
)
@click.option(
    "--date-b", required=True, type=str, help="Second comparison date YYYY-MM-DD"
)
@click.option(
    "--cloud-threshold",
    default=15,
    type=int,
    help="Max local cloud coverage % (default: 15)",
)
@click.option(
    "--change-type",
    type=click.Choice(["vegetation", "water", "urban", "all"]),
    default="all",
    help="Type of change to detect",
)
@click.option(
    "--threshold", default=0.2, type=float, help="Change detection threshold (0.1-1.0)"
)
@click.option(
    "--output", required=True, type=click.Path(), help="Output directory path"
)
@click.option(
    "--name",
    default=None,
    type=str,
    help="Location name for output files (default: lat_lon)",
)
@click.option(
    "--non-interactive",
    is_flag=True,
    help="Skip interactive prompts, use recommended alternatives automatically",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without executing (check dates, cloud coverage only)",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    center: str,
    size: int,
    date_a: str,
    date_b: str,
    cloud_threshold: int,
    change_type: str,
    threshold: float,
    output: str,
    name: Optional[str],
    non_interactive: bool,
    dry_run: bool,
) -> None:
    """
    Execute complete change detection analysis.

    Requires explicit dates (--date-a and --date-b). The tool will check local
    cloud coverage at your area of interest and suggest alternatives if needed.
    """
    gee_client = ctx.obj["gee_client"]
    if not gee_client:
        click.echo("[ERROR] Not authenticated with Google Earth Engine", err=True)
        click.echo("Run 'satchange config init' first", err=True)
        sys.exit(1)

