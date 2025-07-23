#!/usr/bin/env python3
"""
ToxTrac Data Analyzer - Main CLI Application

A comprehensive command-line tool for analyzing animal tracking data from ToxTrac.
This tool provides zone-based analysis, movement statistics, and behavioral metrics
for tracking experiments.

Author: Aman Rathore
Contact: amanr.me | amanrathore9753 <at> gmail <dot> com
Created on: Monday, July 14, 2025 at 10:40
"""

import typer
from pathlib import Path
from typing import List, Optional, Annotated
import sys
import os
import json
from enum import Enum

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.logging_config import (
    setup_logging, get_logger, console, print_success, 
    print_error, print_info, print_header
)
from rich.panel import Panel
from src.data_loader import ToxTracDataLoader
from src.analyzer import ToxTracAnalyzer  
from src.output import ResultsExporter
from src.models import BoundingBox
from src.zone_gui import create_zones_gui_from_data, create_zones_gui_empty, create_zones_gui_with_image


class ZoneType(str, Enum):
    """Zone type options for analysis."""
    epm = "epm"
    open_field = "open-field"
    oft = "oft"
    oft_individual = "oft-individual"
    custom = "custom"


class OutputFormat(str, Enum):
    """Output format options."""
    csv = "csv"
    json = "json"
    console = "console"


class LogLevel(str, Enum):
    """Logging level options."""
    debug = "DEBUG"
    info = "INFO"
    warning = "WARNING"
    error = "ERROR"

# Initialize Typer app
app = typer.Typer(
    name="toxtrac-analyzer",
    help="🐭 ToxTrac Data Analyzer - Analyze animal tracking data from ToxTrac",
    add_completion=False,
    rich_markup_mode="rich"
)

# Global logger (will be initialized in main)
logger = None


@app.command()
def analyze(
    directories: Annotated[
        List[Path],
        typer.Argument(
            help="Directories containing ToxTrac data to analyze",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True
        )
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-o",
            help="Output directory for results (default: current directory)"
        )
    ] = None,
    zone_type: Annotated[
        ZoneType,
        typer.Option(
            "--zones", "-z",
            help="Type of zones to analyze:\n" +
                 "• 'epm' - Standard EPM zones\n" +
                 "• 'open-field' - Open field zones\n" +
                 "• 'oft' - Single OFT zones covering all sessions\n" +
                 "• 'oft-individual' - Separate OFT zones for each RealSpace file\n" +
                 "• 'custom' - Load zones from JSON file"
        )
    ] = ZoneType.epm,
    custom_zones: Annotated[
        Optional[Path],
        typer.Option(
            "--custom-zones-file",
            help="JSON file defining custom zones (required if --zones=custom)"
        )
    ] = None,
    output_format: Annotated[
        List[OutputFormat],
        typer.Option(
            "--format", "-f",
            help="Output format(s)"
        )
    ] = [OutputFormat.console, OutputFormat.csv],
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level", "-l",
            help="Logging level"
        )
    ] = LogLevel.info,
    log_file: Annotated[
        Optional[Path],
        typer.Option(
            "--log-file",
            help="Path to log file (optional)"
        )
    ] = None,
    movement_stats: Annotated[
        bool,
        typer.Option(
            "--movement/--no-movement",
            help="Calculate movement statistics (distance, speed)"
        )
    ] = True,
    individual_reports: Annotated[
        bool,
        typer.Option(
            "--individual/--no-individual",
            help="Show individual session reports in console output"
        )
    ] = False,
    inner_zone_ratio: Annotated[
        float,
        typer.Option(
            "--inner-ratio",
            help="Inner zone size ratio for OFT zones (0.0-1.0). Inner zone will be this fraction of the outer zone size."
        )
    ] = 0.6,
    generate_plots: Annotated[
        bool,
        typer.Option(
            "--plots/--no-plots",
            help="Generate trajectory plots for each session"
        )
    ] = False,
    save_zones: Annotated[
        Optional[Path],
        typer.Option(
            "--save-zones",
            help="Save the generated zones to a JSON file for reuse in other projects"
        )
    ] = None
):
    """
    🔬 Analyze ToxTrac tracking data for time spent in zones and movement patterns.
    
    This command processes ToxTrac tracking files from the specified directories
    and calculates detailed zone occupancy statistics, movement metrics, and 
    behavioral patterns.
    
    Examples:
        # Analyze EPM data from multiple directories
        python main.py analyze data/session1 data/session2 --zones epm
        
        # Analyze with custom output location and JSON format
        python main.py analyze data/ -o results/ -f json csv
        
        # Save generated zones for reuse in other projects
        python main.py analyze data/ --zones oft-individual --save-zones zones/my_zones.json
        
        # Load previously saved zones for analysis
        python main.py analyze data/ --zones custom --custom-zones-file zones/my_zones.json
        
        # Debug mode with detailed logging
        python main.py analyze data/ --log-level DEBUG --log-file debug.log
    """
    global logger
    
    # Setup logging
    logger = setup_logging(
        log_level=log_level,
        log_file=log_file,
        console_output=True
    )
    
    # Display header
    print_header("ToxTrac Data Analyzer")
    
    try:
        # Validate inputs
        if zone_type.lower() == "custom" and not custom_zones:
            print_error("Custom zones file is required when using --zones=custom")
            raise typer.Exit(1)
        
        if not directories:
            print_error("At least one directory must be specified")
            raise typer.Exit(1)
        
        # Set default output directory
        if output_dir is None:
            output_dir = Path.cwd() / "results"
        
        logger.info(f"Starting analysis of {len(directories)} directories")
        logger.info(f"Zone type: {zone_type}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Output formats: {output_format}")
        
        # Initialize components
        data_loader = ToxTracDataLoader()
        analyzer = ToxTracAnalyzer()
        exporter = ResultsExporter()
        
        # Load tracking data
        print_info("Loading tracking data...")
        sessions = data_loader.load_multiple_sessions(directories)
        
        if not sessions:
            print_error("No valid tracking sessions found in the specified directories")
            raise typer.Exit(1)
        
        print_success(f"Loaded {len(sessions)} tracking sessions")
        
        # Create zones
        print_info(f"Creating {zone_type} zones...")
        zones = _create_zones(zone_type, custom_zones, sessions, data_loader, inner_zone_ratio)
        
        if not zones:
            print_error("No zones were created")
            raise typer.Exit(1)
        
        print_success(f"Created {len(zones)} zones: {[z.name for z in zones]}")
        
        # Save zones if requested
        if save_zones:
            print_info(f"Saving zones to {save_zones}...")
            _save_zones_to_file(zones, save_zones)
            print_success(f"Zones saved to {save_zones}")
        
        # Perform analysis
        print_info("Analyzing sessions...")
        results = analyzer.analyze_multiple_sessions(sessions, zones, movement_stats)
        
        if not results:
            print_error("Analysis failed - no results generated")
            raise typer.Exit(1)
        
        print_success(f"Analyzed {len(results)} sessions successfully")
        
        # Output results
        print_info("Generating output...")
        
        saved_files = []
        
        # Create timestamped output directory once and use it for all outputs
        import pandas as pd
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        timestamped_dir = output_dir / f"toxtrac_analysis_{timestamp}"
        
        # Console output
        if "console" in output_format:
            print_header("Analysis Results")
            
            # Summary report
            exporter.display_summary_report(results)
            
            # Individual reports if requested
            if individual_reports:
                exporter.display_individual_sessions(results)
        
        # File outputs
        file_formats = [fmt for fmt in output_format if fmt != "console"]
        if file_formats:
            # Pass the timestamped directory to the exporter
            saved_files = exporter.save_report_to_directory(results, timestamped_dir, file_formats)
        
        # Generate trajectory plots if requested
        if generate_plots:
            print_info("Generating comprehensive plots...")
            # Use the same timestamped directory
            plots_dir = timestamped_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            for session, result in zip(sessions, results):
                session_plots_dir = plots_dir / session.session_name
                session_plots_dir.mkdir(exist_ok=True)
                
                try:
                    # 1. Trajectory plot with zones
                    plot_file = session_plots_dir / f"{session.session_name}_trajectory_zones.png"
                    analyzer.create_trajectory_plot(
                        session=session,
                        zones=zones,
                        output_file=plot_file,
                        show_zones=True,
                        color_by_zone=True,
                        show_speed=False
                    )
                    saved_files.append(plot_file)
                    
                    # 2. Speed-colored trajectory plot
                    speed_plot_file = session_plots_dir / f"{session.session_name}_trajectory_speed.png"
                    analyzer.create_trajectory_plot(
                        session=session,
                        zones=zones,
                        output_file=speed_plot_file,
                        show_zones=True,
                        color_by_zone=False,
                        show_speed=True
                    )
                    saved_files.append(speed_plot_file)
                    
                    # 3. Zone occupancy bar chart
                    occupancy_plot_file = session_plots_dir / f"{session.session_name}_zone_occupancy.png"
                    analyzer.create_zone_occupancy_plot(
                        result=result,
                        output_file=occupancy_plot_file
                    )
                    saved_files.append(occupancy_plot_file)
                    
                    # 4. Speed distribution histogram
                    speed_dist_plot_file = session_plots_dir / f"{session.session_name}_speed_distribution.png"
                    analyzer.create_speed_distribution_plot(
                        session=session,
                        output_file=speed_dist_plot_file
                    )
                    saved_files.append(speed_dist_plot_file)
                    
                    # 5. Movement heatmap
                    heatmap_plot_file = session_plots_dir / f"{session.session_name}_movement_heatmap.png"
                    analyzer.create_movement_heatmap(
                        session=session,
                        zones=zones,
                        output_file=heatmap_plot_file
                    )
                    saved_files.append(heatmap_plot_file)
                    
                    # 6. Time series analysis
                    timeseries_plot_file = session_plots_dir / f"{session.session_name}_timeseries.png"
                    analyzer.create_timeseries_plot(
                        session=session,
                        zones=zones,
                        output_file=timeseries_plot_file
                    )
                    saved_files.append(timeseries_plot_file)
                    
                    print_info(f"Generated 6 plots for session: {session.session_name}")
                    
                except Exception as e:
                    print_error(f"Failed to generate plots for {session.session_name}: {e}")
                    continue
            
            # 7. Summary plots across all sessions
            if len(results) > 1:
                try:
                    summary_plots_dir = plots_dir / "summary"
                    summary_plots_dir.mkdir(exist_ok=True)
                    
                    # Session comparison plot
                    comparison_plot_file = summary_plots_dir / "session_comparison.png"
                    analyzer.create_session_comparison_plot(
                        results=results,
                        output_file=comparison_plot_file
                    )
                    saved_files.append(comparison_plot_file)
                    
                    # Zone usage summary
                    zone_summary_plot_file = summary_plots_dir / "zone_usage_summary.png"
                    analyzer.create_zone_usage_summary_plot(
                        results=results,
                        output_file=zone_summary_plot_file
                    )
                    saved_files.append(zone_summary_plot_file)
                    
                    print_info("Generated 2 summary plots for all sessions")
                    
                except Exception as e:
                    print_error(f"Failed to generate summary plots: {e}")
            
            print_success(f"All plots saved to: {plots_dir}")
        
        # Final summary
        print_header("Analysis Complete")
        print_success(f"Processed {len(sessions)} sessions")
        print_success(f"Analyzed {len(zones)} zones per session")
        
        if saved_files:
            print_info("Saved files:")
            for file_path in saved_files:
                console.print(f"  📄 {file_path}")
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print_error(f"Analysis failed: {e}")
        raise typer.Exit(1)


def _create_zones(
    zone_type: str, 
    custom_zones_file: Optional[Path], 
    sessions: List,
    data_loader: ToxTracDataLoader,
    inner_zone_ratio: float = 0.6
) -> List[BoundingBox]:
    """
    Create zones based on the specified type.
    
    Args:
        zone_type: Type of zones ("epm", "open-field", "oft", "custom")
        custom_zones_file: Path to custom zones file (if applicable)
        sessions: List of tracking sessions for zone creation
        data_loader: Data loader instance
        inner_zone_ratio: Ratio for inner zone size in OFT (0.0-1.0)
        
    Returns:
        List of BoundingBox objects
    """
    if zone_type.lower() == "epm":
        if not sessions:
            raise ValueError("Need at least one session to create EPM zones")
        return data_loader.create_standard_epm_zones_from_multiple_sessions(sessions)
    
    elif zone_type.lower() == "open-field":
        if not sessions:
            raise ValueError("Need at least one session to create Open Field zones")
        return data_loader.create_open_field_zones_from_multiple_sessions(sessions)
    
    elif zone_type.lower() == "oft":
        if not sessions:
            raise ValueError("Need at least one session to create OFT zones")
        return data_loader.create_oft_zones_from_multiple_sessions(sessions, inner_zone_ratio)
    
    elif zone_type.lower() == "oft-individual":
        if not sessions:
            raise ValueError("Need at least one session to create individual OFT zones")
        return data_loader.create_individual_oft_zones_for_each_session(sessions, inner_zone_ratio)
    
    elif zone_type.lower() == "custom":
        if not custom_zones_file or not custom_zones_file.exists():
            raise FileNotFoundError(f"Custom zones file not found: {custom_zones_file}")
        
        return _load_custom_zones(custom_zones_file)
    
    else:
        raise ValueError(f"Unknown zone type: {zone_type}")


def _create_zone_visualization(
    zones: List[BoundingBox],
    tracking_file: Optional[Path] = None,
    output_image: Optional[Path] = None,
    show_labels: bool = True,
    show_coords: bool = False
) -> None:
    """
    Create a visualization of zones with optional tracking data background.
    
    Args:
        zones: List of zone bounding boxes to visualize
        tracking_file: Optional tracking data file for background
        output_image: Optional path to save the image
        show_labels: Whether to show zone labels
        show_coords: Whether to show coordinates in labels
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    
    print_info("Creating zone visualization...")
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Load and plot tracking data if provided
    if tracking_file and tracking_file.exists():
        print_info(f"Loading tracking data from: {tracking_file}")
        try:
            data_loader = ToxTracDataLoader()
            session = data_loader.load_tracking_data(tracking_file)
            df = session.get_dataframe()
            
            # Plot tracking data as background
            ax.scatter(df['x'], df['y'], alpha=0.3, s=1, c='lightblue', 
                      label=f'Tracking data ({len(df)} points)', zorder=1)
            
            ax.set_title(f'Zone Visualization - {session.session_name}')
            
        except Exception as e:
            print_error(f"Failed to load tracking data: {e}")
            ax.set_title('Zone Visualization (no tracking data)')
    else:
        ax.set_title('Zone Visualization')
    
    # Get colors for zones
    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(zones) > len(colors):
        colors = colors * (len(zones) // len(colors) + 1)
    
    # Plot zones
    for i, zone in enumerate(zones):
        color = colors[i % len(colors)]
        
        # Create rectangle
        width = zone.x_max - zone.x_min
        height = zone.y_max - zone.y_min
        
        rect = patches.Rectangle(
            (zone.x_min, zone.y_min), 
            width, 
            height,
            linewidth=2, 
            edgecolor=color, 
            facecolor=color, 
            alpha=0.3,
            label=zone.name,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Add zone label
        if show_labels:
            center_x = zone.x_min + width / 2
            center_y = zone.y_min + height / 2
            
            if show_coords:
                label_text = f"{zone.name}\n({zone.x_min:.1f},{zone.y_min:.1f})\nto\n({zone.x_max:.1f},{zone.y_max:.1f})"
            else:
                label_text = zone.name
            
            ax.text(center_x, center_y, label_text, 
                   ha='center', va='center', fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   zorder=3)
    
    # Set axis properties
    ax.set_xlabel('X Coordinate (mm)')
    ax.set_ylabel('Y Coordinate (mm)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Auto-scale to fit all zones with some padding
    all_x = [zone.x_min for zone in zones] + [zone.x_max for zone in zones]
    all_y = [zone.y_min for zone in zones] + [zone.y_max for zone in zones]
    
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    
    ax.set_xlim(min(all_x) - x_range * 0.1, max(all_x) + x_range * 0.1)
    ax.set_ylim(min(all_y) - y_range * 0.1, max(all_y) + y_range * 0.1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image if requested
    if output_image:
        output_image.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_image, dpi=300, bbox_inches='tight')
        print_success(f"Visualization saved to: {output_image}")
    
    # Show plot
    print_info("Displaying visualization... (close window to continue)")
    print_info(f"Zone Summary:")
    for zone in zones:
        width = zone.x_max - zone.x_min
        height = zone.y_max - zone.y_min
        area = width * height
        print_info(f"  {zone.name}: {width:.1f} × {height:.1f} mm (area: {area:.1f} mm²)")
    
    plt.show()


def _load_custom_zones(zones_file: Path) -> List[BoundingBox]:
    """
    Load custom zones from a JSON file.
    
    Args:
        zones_file: Path to JSON file with zone definitions
        
    Returns:
        List of BoundingBox objects
    """
    import json
    
    with open(zones_file, 'r') as f:
        zones_data = json.load(f)
    
    zones = []
    for zone_data in zones_data:
        zone = BoundingBox(
            name=zone_data['name'],
            x_min=zone_data['x_min'],
            y_min=zone_data['y_min'],
            x_max=zone_data['x_max'],
            y_max=zone_data['y_max'],
            description=zone_data.get('description')
        )
        zones.append(zone)
    
    return zones


def _save_zones_to_file(zones: List[BoundingBox], output_file: Path) -> None:
    """
    Save zones to a JSON file.
    
    Args:
        zones: List of BoundingBox zones to save
        output_file: Path to save the zones file
    """
    import json
    
    zones_data = []
    for zone in zones:
        zones_data.append({
            'name': zone.name,
            'x_min': zone.x_min,
            'y_min': zone.y_min,
            'x_max': zone.x_max,
            'y_max': zone.y_max,
            'description': zone.description
        })
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(zones_data, f, indent=2)
    
    # Print zone summary
    print_info("Zone Summary:")
    for zone in zones:
        print_info(f"  {zone.name}: ({zone.x_min:.1f}, {zone.y_min:.1f}) to ({zone.x_max:.1f}, {zone.y_max:.1f})")


@app.command()
def create_zones_template(
    output_file: Annotated[
        Path,
        typer.Argument(help="Output file for the zones template")
    ]
):
    """
    📝 Create a template file for defining custom zones.
    
    This command generates a JSON template file that can be used to define
    custom bounding box zones for analysis.
    
    Example:
        python main.py create-zones-template custom_zones.json
    """
    template = [
        {
            "name": "center",
            "x_min": 0.0,
            "y_min": 0.0,
            "x_max": 100.0,
            "y_max": 100.0,
            "description": "Center zone"
        },
        {
            "name": "periphery",
            "x_min": -50.0,
            "y_min": -50.0,
            "x_max": 150.0,
            "y_max": 150.0,
            "description": "Peripheral zone"
        }
    ]
    
    import json
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print_success(f"Created zones template: {output_file}")
    print_info("Edit the template file to define your custom zones")


@app.command()
def list_files(
    directories: Annotated[
        List[Path],
        typer.Argument(
            help="Directories to scan for ToxTrac files",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True
        )
    ]
):
    """
    📁 List all ToxTrac tracking files in the specified directories.
    
    This command scans directories and reports all detected ToxTrac tracking
    files without performing analysis.
    
    Example:
        python main.py list-files data/session1 data/session2
    """
    print_header("ToxTrac File Scanner")
    
    data_loader = ToxTracDataLoader()
    
    total_files = 0
    for directory in directories:
        print_info(f"Scanning: {directory}")
        
        tracking_files = data_loader.find_tracking_files(directory)
        
        if tracking_files:
            console.print(f"  Found {len(tracking_files)} files:")
            for file_path in tracking_files:
                console.print(f"    📄 {file_path.relative_to(directory)}")
            total_files += len(tracking_files)
        else:
            console.print("    ❌ No tracking files found")
    
    print_success(f"Total files found: {total_files}")


@app.command()
def create_zones_gui(
    output_file: Annotated[
        Path,
        typer.Argument(help="Output file for the zone definitions (JSON)")
    ],
    tracking_file: Annotated[
        Optional[Path],
        typer.Option(
            "--tracking-file", "-t",
            help="Tracking file to use as background (optional)"
        )
    ] = None,
    background_image: Annotated[
        Optional[Path],
        typer.Option(
            "--background-image", "-b",
            help="Background image file for reference (JPG, PNG, etc.)"
        )
    ] = None,
    image_coords: Annotated[
        Optional[str],
        typer.Option(
            "--image-coords",
            help="Image coordinate bounds as 'x_min,y_min,x_max,y_max' (required with --background-image)"
        )
    ] = None,
    x_range: Annotated[
        str,
        typer.Option(
            "--x-range",
            help="X coordinate range for empty plot (format: 'min,max')"
        )
    ] = "-200,200",
    y_range: Annotated[
        str,
        typer.Option(
            "--y-range", 
            help="Y coordinate range for empty plot (format: 'min,max')"
        )
    ] = "-200,200"
):
    """
    🎯 Open interactive GUI to create zones by clicking on plots.
    
    This command opens a matplotlib-based GUI where you can visually define
    zones by clicking on tracking data plots, background images, or empty coordinate systems.
    
    Features:
    - Click to define rectangular zones
    - Use existing tracking data as background
    - Load reference images (arena photos, etc.)
    - Name zones interactively
    - Save to JSON format for use with --zones=custom
    
    Examples:
        # Create zones with tracking data background
        python main.py create-zones-gui zones.json -t data/Tracking_RealSpace.txt
        
        # Create zones with background image reference
        python main.py create-zones-gui zones.json -b arena_photo.jpg --image-coords="-150,-100,150,100"
        
        # Create zones on empty coordinate system
        python main.py create-zones-gui zones.json --x-range=-100,100 --y-range=-100,100
        
        # Use the created zones file
        python main.py analyze data/ --zones=custom --custom-zones-file=zones.json
    """
    print_header("Zone Creator GUI")
    
    try:
        # Validate inputs
        if background_image and not image_coords:
            print_error("--image-coords is required when using --background-image")
            print_info("Format: --image-coords='x_min,y_min,x_max,y_max'")
            raise typer.Exit(1)
        
        if background_image and tracking_file:
            print_error("Cannot use both --background-image and --tracking-file simultaneously")
            raise typer.Exit(1)
        
        # Parse coordinate ranges
        try:
            x_min, x_max = map(float, x_range.split(','))
            y_min, y_max = map(float, y_range.split(','))
        except ValueError:
            print_error("Invalid coordinate range format. Use 'min,max' (e.g., '-100,100')")
            raise typer.Exit(1)
        
        if tracking_file:
            # GUI with tracking data background
            if not tracking_file.exists():
                print_error(f"Tracking file not found: {tracking_file}")
                raise typer.Exit(1)
            
            print_info(f"Opening GUI with tracking data: {tracking_file}")
            create_zones_gui_from_data(tracking_file, output_file)
            
        elif background_image:
            # GUI with background image
            if not background_image.exists():
                print_error(f"Background image not found: {background_image}")
                raise typer.Exit(1)
            
            try:
                img_x_min, img_y_min, img_x_max, img_y_max = map(float, image_coords.split(','))
            except ValueError:
                print_error("Invalid image-coords format. Use 'x_min,y_min,x_max,y_max'")
                raise typer.Exit(1)
            
            print_info(f"Opening GUI with background image: {background_image}")
            print_info(f"Image coordinates: ({img_x_min}, {img_y_min}) to ({img_x_max}, {img_y_max})")
            create_zones_gui_with_image(background_image, (img_x_min, img_y_min, img_x_max, img_y_max), output_file)
            
        else:
            # GUI with empty coordinate system
            print_info(f"Opening GUI with coordinate range: X({x_min}, {x_max}), Y({y_min}, {y_max})")
            create_zones_gui_empty(output_file, (x_min, x_max), (y_min, y_max))
            
    except Exception as e:
        logger_local = get_logger("main")
        logger_local.error(f"GUI creation failed: {e}")
        print_error(f"GUI creation failed: {e}")
        raise typer.Exit(1)


@app.command()
def visualize_zones(
    zones_file: Annotated[
        Path,
        typer.Argument(
            help="JSON file containing zone definitions",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True
        )
    ],
    tracking_file: Annotated[
        Optional[Path],
        typer.Option(
            "--tracking-file", "-t",
            help="Tracking file to use as background (optional)"
        )
    ] = None,
    output_image: Annotated[
        Optional[Path],
        typer.Option(
            "--save-image", "-s",
            help="Save visualization as image file (PNG/PDF/SVG)"
        )
    ] = None,
    show_labels: Annotated[
        bool,
        typer.Option(
            "--labels/--no-labels",
            help="Show zone labels on the plot"
        )
    ] = True,
    show_coords: Annotated[
        bool,
        typer.Option(
            "--coords/--no-coords",
            help="Show zone coordinates in labels"
        )
    ] = False
):
    """
    📊 Visualize zones from a JSON file with optional tracking data background.
    
    This command creates a matplotlib plot showing the defined zones, optionally
    overlaid on tracking data for context. Useful for validating zone definitions
    and understanding spatial relationships.
    
    Examples:
        # Visualize zones only
        python main.py visualize-zones custom_zones.json
        
        # Visualize zones with tracking data background
        python main.py visualize-zones custom_zones.json -t data/Tracking_RealSpace.txt
        
        # Save visualization as image
        python main.py visualize-zones custom_zones.json -s zones_plot.png
        
        # Show coordinates in labels
        python main.py visualize-zones custom_zones.json --coords
    """
    print_header("Zone Visualizer")
    
    try:
        # Load zones
        print_info(f"Loading zones from: {zones_file}")
        zones = _load_custom_zones(zones_file)
        
        if not zones:
            print_error("No zones found in the file")
            raise typer.Exit(1)
        
        print_success(f"Loaded {len(zones)} zones: {[z.name for z in zones]}")
        
        # Create visualization
        _create_zone_visualization(
            zones, 
            tracking_file, 
            output_image, 
            show_labels, 
            show_coords
        )
        
    except Exception as e:
        print_error(f"Visualization failed: {e}")
        raise typer.Exit(1)


@app.command()
def plot_trajectory(
    tracking_file: Annotated[
        Path,
        typer.Argument(
            help="Tracking data file to plot",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True
        )
    ],
    zones_file: Annotated[
        Optional[Path],
        typer.Option(
            "--zones-file", "-z",
            help="JSON file containing zone definitions"
        )
    ] = None,
    output_image: Annotated[
        Optional[Path],
        typer.Option(
            "--save-image", "-s",
            help="Save trajectory plot as image file (PNG/PDF/SVG)"
        )
    ] = None,
    color_by_zone: Annotated[
        bool,
        typer.Option(
            "--color-zones/--no-color-zones",
            help="Color trajectory points by current zone"
        )
    ] = True,
    show_speed: Annotated[
        bool,
        typer.Option(
            "--show-speed/--no-speed",
            help="Color trajectory by speed instead of zones"
        )
    ] = False,
    show_zones: Annotated[
        bool,
        typer.Option(
            "--show-zone-boundaries/--no-zone-boundaries",
            help="Show zone boundary lines"
        )
    ] = True
):
    """
    🗺️ Create trajectory plots showing animal movement paths with zones.
    
    This command creates detailed trajectory visualizations showing the path taken
    by the animal, optionally colored by zones or speed, with zone boundaries overlaid.
    
    Examples:
        # Basic trajectory plot
        python main.py plot-trajectory data/Tracking_RealSpace.txt
        
        # Trajectory with zones overlay
        python main.py plot-trajectory data/Tracking_RealSpace.txt -z custom_zones.json
        
        # Save trajectory plot to file
        python main.py plot-trajectory data/Tracking_RealSpace.txt -s trajectory.png
        
        # Show speed-colored trajectory
        python main.py plot-trajectory data/Tracking_RealSpace.txt --show-speed
    """
    print_header("Trajectory Plotter")
    
    try:
        # Load tracking data
        print_info(f"Loading tracking data from: {tracking_file}")
        data_loader = ToxTracDataLoader()
        session = data_loader.load_tracking_data(tracking_file)
        
        print_success(f"Loaded session: {session.session_name}")
        print_info(f"Duration: {session.total_duration:.1f} seconds")
        print_info(f"Data points: {len(session.data_points)}")
        
        # Load zones if provided
        zones = []
        if zones_file and zones_file.exists():
            print_info(f"Loading zones from: {zones_file}")
            zones = _load_custom_zones(zones_file)
            print_success(f"Loaded {len(zones)} zones: {[z.name for z in zones]}")
        
        # Create trajectory plot
        analyzer = ToxTracAnalyzer()
        fig = analyzer.create_trajectory_plot(
            session=session,
            zones=zones,
            output_file=output_image,
            show_zones=show_zones,
            color_by_zone=color_by_zone and not show_speed,
            show_speed=show_speed
        )
        
        print_success("Trajectory plot created successfully!")
        
        # Show plot
        import matplotlib.pyplot as plt
        print_info("Displaying trajectory plot... (close window to continue)")
        plt.show()
        
    except Exception as e:
        print_error(f"Failed to create trajectory plot: {e}")
        raise typer.Exit(1)


@app.command()
def version():
    """📋 Show version information."""
    from src import __version__, __author__, __email__
    
    console.print(Panel.fit(
        f"[bold]ToxTrac Data Analyzer[/bold]\n"
        f"Version: {__version__}\n"
        f"Author: {__author__}\n"
        f"Contact: {__email__}",
        title="Version Information",
        border_style="blue"
    ))


if __name__ == "__main__":
    app()
