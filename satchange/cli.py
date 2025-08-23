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

    try:
        # Parse inputs
        lat, lon = parse_coordinates(center)

        # Validate parameters
        from .utils import validate_pixel_size, validate_threshold

        validate_pixel_size(size)
        validate_threshold(threshold, 0.1, 1.0)

        # Create output directory
        os.makedirs(output, exist_ok=True)

        click.echo("Starting change detection analysis...")
        click.echo(f"  Location: {lat}, {lon}")
        click.echo(f"  Area: {size}x{size} pixels")
        click.echo(f"  Requested dates: {date_a} and {date_b}")
        click.echo(f"  Cloud threshold: {cloud_threshold}%")
        click.echo(f"  Output: {output}")
        click.echo(f"  Change type: {change_type}")
        click.echo(f"  Detection threshold: {threshold}")

        # ========================================
        # STEP 1: Check local cloud coverage
        # ========================================
        click.echo("\n[Step 1/4] Checking local cloud coverage...")

        center_coords = (lat, lon)

        # Check Date A
        with spinner(f"Checking cloud coverage for {date_a}..."):
            check_a = gee_client.check_local_cloud(date_a, center_coords, size)
        display_cloud_check_result(check_a, "Date A")

        # Check Date B
        with spinner(f"Checking cloud coverage for {date_b}..."):
            check_b = gee_client.check_local_cloud(date_b, center_coords, size)
        display_cloud_check_result(check_b, "Date B")

        # Track final dates to use
        final_date_a = date_a
        final_date_b = date_b
        final_image_id_a = check_a.get("image_id")
        final_image_id_b = check_b.get("image_id")

        # ========================================
        # STEP 2: Find alternatives if needed
        # ========================================
        needs_alternatives = False

        # Check if Date A needs alternatives
        if not check_a["found"] or not check_a["is_good"]:
            needs_alternatives = True
            click.echo(f"\n[Step 2/4] Finding alternatives for Date A ({date_a})...")

            with spinner("Searching for alternative dates..."):
                alt_result_a = gee_client.find_alternative_dates(
                    date_a, center_coords, cloud_threshold, size
                )

            if alt_result_a["threshold_met"]:
                click.echo(
                    f"  Found good alternatives in {alt_result_a['search_window']}"
                )
            else:
                click.echo(
                    f"  [WARN] No alternatives below {cloud_threshold}% within ±3 months"
                )

            display_alternatives(alt_result_a["alternatives"])

            if non_interactive:
                # Auto-select recommended
                recommended = next(
                    (a for a in alt_result_a["alternatives"] if a["is_recommended"]),
                    None,
                )
                if recommended:
                    final_date_a = recommended["date"]
                    final_image_id_a = recommended["image_id"]
                    click.echo(f"  Auto-selected: {final_date_a}")
            else:
                # Interactive selection
                final_date_a = prompt_date_selection(
                    alt_result_a["alternatives"], date_a, "Date A"
                )
                selected_alt = next(
                    (
                        a
                        for a in alt_result_a["alternatives"]
                        if a["date"] == final_date_a
                    ),
                    None,
                )
                if selected_alt:
                    final_image_id_a = selected_alt["image_id"]

        # Check if Date B needs alternatives
        if not check_b["found"] or not check_b["is_good"]:
            needs_alternatives = True
            click.echo(f"\n[Step 2/4] Finding alternatives for Date B ({date_b})...")

            with spinner("Searching for alternative dates..."):
                alt_result_b = gee_client.find_alternative_dates(
                    date_b, center_coords, cloud_threshold, size
                )

            if alt_result_b["threshold_met"]:
                click.echo(
                    f"  Found good alternatives in {alt_result_b['search_window']}"
                )
            else:
                click.echo(
                    f"  [WARN] No alternatives below {cloud_threshold}% within ±3 months"
                )

            display_alternatives(alt_result_b["alternatives"])

            if non_interactive:
                # Auto-select recommended
                recommended = next(
                    (a for a in alt_result_b["alternatives"] if a["is_recommended"]),
                    None,
                )
                if recommended:
                    final_date_b = recommended["date"]
                    final_image_id_b = recommended["image_id"]
                    click.echo(f"  Auto-selected: {final_date_b}")
            else:
                # Interactive selection
                final_date_b = prompt_date_selection(
                    alt_result_b["alternatives"], date_b, "Date B"
                )
                selected_alt = next(
                    (
                        a
                        for a in alt_result_b["alternatives"]
                        if a["date"] == final_date_b
                    ),
                    None,
                )
                if selected_alt:
                    final_image_id_b = selected_alt["image_id"]

        if not needs_alternatives:
            click.echo(
                "\n[Step 2/4] Both dates have good cloud coverage - no alternatives needed"
            )

        # Validate we have image IDs
        if not final_image_id_a or not final_image_id_b:
            click.echo(
                "[ERROR] Could not find valid images for the selected dates", err=True
            )
            sys.exit(1)

        click.echo(f"\n  Final dates: {final_date_a} -> {final_date_b}")

        # Generate output filename prefix
        output_prefix = generate_output_prefix(
            name, lat, lon, final_date_a, final_date_b
        )
        click.echo(f"  Output prefix: {output_prefix}")

        # Dry-run: report what would be done and exit
        if dry_run:
            click.echo("\n[DRY RUN] Would execute the following:")
            click.echo(f"  Download images for dates: {final_date_a}, {final_date_b}")
            click.echo(f"  Area: {size}x{size} pixels at ({lat}, {lon})")
            click.echo(f"  Change type: {change_type}")
            click.echo(f"  Detection threshold: {threshold}")
            click.echo(f"  Output directory: {output}")
            click.echo("\n[DRY RUN] No images downloaded, no analysis performed.")
            return

        # Check disk space before downloading
        from .utils import check_disk_space

        disk_check = check_disk_space(output, required_mb=50.0)
        if not disk_check["sufficient"]:
            click.echo(
                f"[ERROR] Insufficient disk space: {disk_check['available_mb']:.0f}MB "
                f"available, {disk_check['required_mb']:.0f}MB required",
                err=True,
            )
            sys.exit(1)

        # ========================================
        # STEP 3: Download and process images
        # ========================================
        click.echo("\n[Step 3/4] Downloading imagery...")

        # Initialize components
        cache_manager = CacheManager(ctx.obj["config"])
        image_processor = ImageProcessor(ctx.obj["config"])
        change_detector = ChangeDetector(threshold=threshold)
        emboss_intensity = 1.0  # Default emboss intensity
        visualization_manager = VisualizationManager(emboss_intensity=emboss_intensity)

        # Create bounding box
        bbox = gee_client.create_bbox(lat, lon, size)

        # Create ee.Image objects from IDs
        import ee

        img_a = ee.Image(final_image_id_a)
        img_b = ee.Image(final_image_id_b)

        # Get image metadata
        img_a_info = gee_client.get_image_info(img_a)
        img_b_info = gee_client.get_image_info(img_b)

        click.echo(
            f"Date A: {img_a_info.get('date', 'Unknown')} (Cloud cover: {img_a_info.get('cloud_coverage', 0):.1f}%)"
        )
        click.echo(
            f"Date B: {img_b_info.get('date', 'Unknown')} (Cloud cover: {img_b_info.get('cloud_coverage', 0):.1f}%)"
        )

        # Define bands to download
        bands = [
            "B4",
            "B3",
            "B2",
            "B8",
            "B11",
            "QA60",
        ]  # RGB (B4, B3, B2) + NIR (B8) + SWIR (B11) + QA

        # Download images with caching
        def download_image_func(image, image_info):
            band_arrays, metadata = gee_client.download_image(image, bbox, bands)
            return {"arrays": band_arrays, "metadata": metadata}

        # Convert final dates to datetime for cache
        from datetime import datetime as dt

        final_date_a_dt = dt.strptime(final_date_a, "%Y-%m-%d")
        final_date_b_dt = dt.strptime(final_date_b, "%Y-%m-%d")

        with progress_bar("Downloading imagery", total=2) as advance:
            data_a, cache_hit_a = cache_manager.get_image_with_cache(
                lat,
                lon,
                size,
                final_date_a_dt,
                bands,
                download_image_func,
                img_a,
                img_a_info,
            )
            band_arrays_a = data_a["arrays"]
            metadata_a = data_a["metadata"]
            advance(1)

            data_b, cache_hit_b = cache_manager.get_image_with_cache(
                lat,
                lon,
                size,
                final_date_b_dt,
                bands,
                download_image_func,
                img_b,
                img_b_info,
            )
            band_arrays_b = data_b["arrays"]
            metadata_b = data_b["metadata"]
            advance(1)

        # Show cache statistics
        cache_stats = cache_manager.get_cache_stats()
        click.echo(f"Cache hit rate: {cache_stats.get('hit_rate', 0):.1f}%")

        # Preprocess images
        with spinner("Preprocessing images..."):
            band_arrays_a, band_arrays_b = image_processor.preprocess_image_pair(
                band_arrays_a, band_arrays_b, metadata_a, metadata_b
            )

        # Get processing summary
        processing_summary = image_processor.get_processing_summary(
            band_arrays_a, band_arrays_b, metadata_a, metadata_b
        )

        if processing_summary["processing_successful"]:
            click.echo("[OK] Image preprocessing completed successfully")

            # Display warnings
            if processing_summary["warnings"]:
                click.echo("\nWarnings:")
                for warning in processing_summary["warnings"]:
                    click.echo(f"  - {warning}")
        else:
            click.echo("[ERROR] Image preprocessing failed", err=True)
            if "error" in processing_summary:
                click.echo(f"Error: {processing_summary['error']}", err=True)
            sys.exit(1)

        # ========================================
        # STEP 4: Run change detection
        # ========================================
        click.echo("\n[Step 4/4] Running change detection...")
        with spinner("Analyzing spectral changes..."):
            change_summary = change_detector.get_change_summary(
                band_arrays_a, band_arrays_b, change_type
            )

        # Get classification and statistics
        if change_type == "all":
            classification = change_summary["classification"]
            change_stats = change_summary["statistics"]
        else:
            # For single change types, create a simple classification
            classification = np.zeros_like(band_arrays_a["B4"], dtype=np.uint8)
            change_results = change_summary["change_results"]

            if change_type == "vegetation":
                classification[change_results["growth_mask"]] = 1
                classification[change_results["loss_mask"]] = 2
            elif change_type == "water":
                classification[change_results["expansion_mask"]] = 3
                classification[change_results["reduction_mask"]] = 4
            elif change_type == "urban":
                classification[change_results["development_mask"]] = 5
                classification[change_results["decline_mask"]] = 6

            # Compute statistics for the specific change type
            change_stats = change_summary["statistics"]
            change_stats["total_change"] = {
                "pixels": change_stats["changed_pixels"],
                "percent": change_stats["change_percentage"],
                "area_km2": round((change_stats["changed_pixels"] * 100) / 1e6, 4),
            }

        # Save intermediate results
        click.echo("Saving analysis results...")

        # Save band arrays
        np.save(os.path.join(output, f"{output_prefix}_bands_a.npy"), band_arrays_a)
        np.save(os.path.join(output, f"{output_prefix}_bands_b.npy"), band_arrays_b)

        # Save classification
        np.save(
            os.path.join(output, f"{output_prefix}_classification.npy"), classification
        )

        # Save change statistics
        with open(os.path.join(output, f"{output_prefix}_change_stats.json"), "w") as f:
            json.dump(change_stats, f, indent=2, cls=NumpyJSONEncoder)

        # Save metadata
        with open(os.path.join(output, f"{output_prefix}_metadata.json"), "w") as f:
            json.dump(
                {
                    "date_a": img_a_info,
                    "date_b": img_b_info,
                    "processing_summary": processing_summary,
                    "cache_stats": cache_stats,
                    "change_detection": {
                        "change_type": change_type,
                        "threshold": threshold,
                        "summary": change_summary.get("summary", ""),
                    },
                    "center_lat": lat,
                    "center_lon": lon,
                    "location_name": name if name else format_location_name(lat, lon),
                    "output_prefix": output_prefix,
                },
                f,
                indent=2,
                cls=NumpyJSONEncoder,
            )

        # Display results
        click.echo(f"\n[OK] Change detection completed successfully!")
        click.echo(f"Results saved to: {output}")
        click.echo(f"Cache usage: {cache_stats.get('usage_percent', 0):.1f}%")

        # Display change summary
        if change_type == "all":
            click.echo(f"\n=== Change Detection Results ===")
            click.echo(
                f"Total area changed: {change_stats['total_change']['percent']}%"
            )
            click.echo(
                f"Vegetation changes: {change_stats['change_types']['vegetation']['percent']}%"
            )
            click.echo(
                f"Water changes: {change_stats['change_types']['water']['percent']}%"
            )
            click.echo(
                f"Urban changes: {change_stats['change_types']['urban']['percent']}%"
            )
        else:
            click.echo(f"\n=== {change_type.title()} Change Detection Results ===")
            click.echo(
                f"Total {change_type} changes: {change_stats['change_percentage']}%"
            )
            if change_type == "vegetation":
                click.echo(f"Vegetation growth: {change_stats['growth_pixels']} pixels")
                click.echo(f"Vegetation loss: {change_stats['loss_pixels']} pixels")
            elif change_type == "water":
                click.echo(
                    f"Water expansion: {change_stats['expansion_pixels']} pixels"
                )
                click.echo(
                    f"Water reduction: {change_stats['reduction_pixels']} pixels"
                )
            elif change_type == "urban":
                click.echo(
                    f"Urban development: {change_stats.get('development_pixels', 0)} pixels"
                )
                click.echo(
                    f"Urban decline: {change_stats.get('decline_pixels', 0)} pixels"
                )

        # Generate visualizations
        click.echo("\nGenerating visualizations...")
        try:
            # Load saved results for visualization
            classification = np.load(
                os.path.join(output, f"{output_prefix}_classification.npy")
            )
            bands_a = np.load(
                os.path.join(output, f"{output_prefix}_bands_a.npy"), allow_pickle=True
            ).item()
            bands_b = np.load(
                os.path.join(output, f"{output_prefix}_bands_b.npy"), allow_pickle=True
            ).item()

            with open(
                os.path.join(output, f"{output_prefix}_change_stats.json"), "r"
            ) as f:
                vis_stats = json.load(f)

            with open(os.path.join(output, f"{output_prefix}_metadata.json"), "r") as f:
                metadata = json.load(f)

            # Generate visualizations with new naming convention
            with spinner("Generating visualizations..."):
                output_files = visualization_manager.generate_all_outputs(
                    bands_a,
                    bands_b,
                    classification,
                    vis_stats,
                    metadata,
                    lat,
                    lon,
                    output,
                    ["static", "interactive", "geotiff"],
                    output_prefix=output_prefix,
                )

            click.echo("[OK] Visualizations generated:")
            for format_type, file_path in output_files.items():
                click.echo(f"  {format_type}: {file_path}")

        except Exception as e:
            click.echo(f"⚠ Visualization generation failed: {e}", err=True)
            click.echo(
                "You can still generate visualizations later using 'satchange export'"
            )

        # Cleanup
        cache_manager.close()

    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(130)
    except (QuotaExceededError, RateLimitError) as e:
        click.echo(f"[ERROR] GEE limit reached: {e}", err=True)
        click.echo("Try again later or reduce the analysis area.", err=True)
        sys.exit(1)
    except NoImageryError as e:
        click.echo(f"[ERROR] No suitable imagery: {e}", err=True)
        click.echo("Try different dates or a larger cloud threshold.", err=True)
        sys.exit(1)
    except DownloadError as e:
        click.echo(f"[ERROR] Download failed: {e}", err=True)
        click.echo("Check your internet connection and try again.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"[ERROR] Analysis failed: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--result",
    required=True,
    type=click.Path(),
    help="Path to analysis result directory",
)
@click.option(
    "--format",
    type=click.Choice(["static", "interactive", "geotiff", "all"]),
    default="all",
    help="Output format",
)
@click.option(
    "--emboss-intensity", default=1.0, type=float, help="Emboss effect strength (0-2)"
)
@click.option(
    "--name",
    default=None,
    type=str,
    help="Location name for output files (default: uses existing or lat_lon)",
)
@click.pass_context
def export(
    ctx: click.Context,
    result: str,
    format: str,
    emboss_intensity: float,
    name: Optional[str],
) -> None:
    """
    Generate visualization outputs from analysis results.
    """
    try:
        # Check if result directory exists
        if not os.path.exists(result):
            click.echo(f"[ERROR] Result directory not found: {result}", err=True)
            sys.exit(1)

        click.echo("Starting visualization generation...")
        click.echo(f"  Result directory: {result}")
        click.echo(f"  Output format: {format}")
        click.echo(f"  Emboss intensity: {emboss_intensity}")

        # Initialize visualization manager
        visualization_manager = VisualizationManager(emboss_intensity=emboss_intensity)

        # Try to find files with new naming convention first, fall back to legacy names
        import glob as glob_module
        import numpy as np
        import json

        # Look for metadata file to get output prefix
        metadata_files = glob_module.glob(os.path.join(result, "*_metadata.json"))
        legacy_metadata = os.path.join(result, "metadata.json")

        output_prefix = None
        metadata_path = None

        if metadata_files:
            # Use new naming convention
            metadata_path = metadata_files[0]
            # Extract prefix from filename (e.g., "loc_2020-01-01_2020-12-31_metadata.json" -> "loc_2020-01-01_2020-12-31")
            basename = os.path.basename(metadata_path)
            output_prefix = basename.replace("_metadata.json", "")
        elif os.path.exists(legacy_metadata):
            # Fall back to legacy naming
            metadata_path = legacy_metadata
            output_prefix = None
        else:
            click.echo("[ERROR] No metadata file found in result directory", err=True)
            sys.exit(1)

        # Define file paths based on naming convention
        if output_prefix:
            classification_path = os.path.join(
                result, f"{output_prefix}_classification.npy"
            )
            bands_a_path = os.path.join(result, f"{output_prefix}_bands_a.npy")
            bands_b_path = os.path.join(result, f"{output_prefix}_bands_b.npy")
            stats_path = os.path.join(result, f"{output_prefix}_change_stats.json")
        else:
            classification_path = os.path.join(result, "classification.npy")
            bands_a_path = os.path.join(result, "bands_a.npy")
            bands_b_path = os.path.join(result, "bands_b.npy")
            stats_path = os.path.join(result, "change_stats.json")

        # Check required files exist
        required_files = [
            classification_path,
            bands_a_path,
            bands_b_path,
            stats_path,
            metadata_path,
        ]
        missing_files = [f for f in required_files if not os.path.exists(f)]

        if missing_files:
            click.echo(
                f"[ERROR] Missing required files: {', '.join(missing_files)}", err=True
            )
            click.echo("Make sure you have run 'satchange analyze' first", err=True)
            sys.exit(1)

        # Load analysis results
        click.echo("Loading analysis results...")

        classification = np.load(classification_path, allow_pickle=True)
        bands_a = np.load(bands_a_path, allow_pickle=True).item()
        bands_b = np.load(bands_b_path, allow_pickle=True).item()

        with open(stats_path, "r") as f:
            stats = json.load(f)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Extract center coordinates from metadata
        center_lat = metadata.get("center_lat", 0.0)
        center_lon = metadata.get("center_lon", 0.0)

        # Determine output prefix for visualization files
        # Priority: --name option > metadata output_prefix > generate from coordinates
        if name:
            # User provided a name, generate new prefix
            date_a = metadata.get("date_a", {}).get("date", "").split("T")[0]
            date_b = metadata.get("date_b", {}).get("date", "").split("T")[0]
            final_output_prefix = generate_output_prefix(
                name, center_lat, center_lon, date_a, date_b
            )
        elif output_prefix:
            # Use existing prefix from files
            final_output_prefix = output_prefix
        else:
            # Generate prefix from metadata (legacy files)
            date_a = metadata.get("date_a", {}).get("date", "").split("T")[0]
            date_b = metadata.get("date_b", {}).get("date", "").split("T")[0]
            final_output_prefix = generate_output_prefix(
                None, center_lat, center_lon, date_a, date_b
            )

        click.echo(f"  Output prefix: {final_output_prefix}")

        # Generate visualizations
        click.echo("Generating visualizations...")

        formats_to_generate = (
            [format] if format != "all" else ["static", "interactive", "geotiff"]
        )
        output_files = visualization_manager.generate_all_outputs(
            bands_a,
            bands_b,
            classification,
            stats,
            metadata,
            center_lat,
            center_lon,
            result,
            formats_to_generate,
            output_prefix=final_output_prefix,
        )

        click.echo("[OK] Visualization generation completed!")
        click.echo("Generated files:")
        for format_type, file_path in output_files.items():
            click.echo(f"  {format_type}: {file_path}")

    except Exception as e:
        click.echo(f"[ERROR] Export failed: {e}", err=True)
        sys.exit(1)


@main.group()
def cache() -> None:
    """Manage local tile storage."""
    pass


@cache.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show cache statistics."""
    try:
        cache_manager = CacheManager(ctx.obj["config"])
        stats = cache_manager.get_cache_stats()

        click.echo("Cache Statistics:")
        click.echo(f"  Total items: {stats.get('total_items', 0)}")
        click.echo(f"  Size: {stats.get('size_formatted', '0 B')}")
        click.echo(f"  Usage: {stats.get('usage_percent', 0):.1f}%")
        click.echo(f"  Hit rate: {stats.get('hit_rate', 0):.1f}%")
        click.echo(f"  Hits: {stats.get('hits', 0)}")
        click.echo(f"  Misses: {stats.get('misses', 0)}")
        click.echo(f"  Evictions: {stats.get('evictions', 0)}")

        cache_manager.close()

    except Exception as e:
        click.echo(f"[ERROR] Failed to get cache status: {e}", err=True)


@cache.command()
@click.pass_context
def clear(ctx: click.Context) -> None:
    """Clear all cached tiles."""
    try:
        cache_manager = CacheManager(ctx.obj["config"])

        if click.confirm("Are you sure you want to clear all cached tiles?"):
            if cache_manager.clear_cache():
                click.echo("[OK] Cache cleared successfully")
            else:
                click.echo("[ERROR] Failed to clear cache", err=True)

        cache_manager.close()

    except Exception as e:
        click.echo(f"[ERROR] Failed to clear cache: {e}", err=True)


@cache.command()
@click.pass_context
def cleanup(ctx: click.Context) -> None:
    """Remove expired cache entries (older than 30 days)."""
    try:
        cache_manager = CacheManager(ctx.obj["config"])
        if cache_manager.cleanup_cache():
            click.echo("[OK] Cache cleanup completed")
        else:
            click.echo("[ERROR] Cache cleanup failed", err=True)
        cache_manager.close()
    except Exception as e:
        click.echo(f"[ERROR] Cache cleanup failed: {e}", err=True)


if __name__ == "__main__":
    main()
