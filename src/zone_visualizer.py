"""
Zone visualization utilities for ToxTrac data analysis.

This module provides functions to visualize zone definitions from JSON files,
with optional tracking data as background.

Author: Aman Rathore
Contact: amanr.me | amanrathore9753 <at> gmail <dot> com
Created on: Monday, July 14, 2025 at 10:40
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import pandas as pd
from pathlib import Path
from typing import List, Optional

from .models import BoundingBox, TrackingSession
from .data_loader import ToxTracDataLoader
from .logging_config import get_logger, print_success, print_error, print_info


def load_zones_from_file(zones_file: Path) -> List[BoundingBox]:
    """
    Load zones from a JSON file.
    
    Args:
        zones_file: Path to JSON file with zone definitions
        
    Returns:
        List of BoundingBox objects
    """
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


def visualize_zones(
    zones: List[BoundingBox],
    tracking_file: Optional[Path] = None,
    output_image: Optional[Path] = None,
    show_labels: bool = True,
    show_coords: bool = False,
    figsize: tuple = (12, 10),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Create a visualization of zones with optional tracking data background.
    
    Args:
        zones: List of zone bounding boxes to visualize
        tracking_file: Optional tracking data file for background
        output_image: Optional path to save the image
        show_labels: Whether to show zone labels
        show_coords: Whether to show coordinates in labels
        figsize: Figure size as (width, height)
        title: Custom title for the plot
        
    Returns:
        matplotlib Figure object
    """
    logger = get_logger("zone_visualizer")
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Load and plot tracking data if provided
    if tracking_file and tracking_file.exists():
        try:
            data_loader = ToxTracDataLoader()
            session = data_loader.load_tracking_data(tracking_file)
            df = session.get_dataframe()
            
            # Plot tracking data as background
            ax.scatter(df['x'], df['y'], alpha=0.3, s=1, c='lightblue', 
                      label=f'Tracking data ({len(df)} points)', zorder=1)
            
            if not title:
                title = f'Zone Visualization - {session.session_name}'
            
        except Exception as e:
            logger.error(f"Failed to load tracking data: {e}")
            if not title:
                title = 'Zone Visualization (no tracking data)'
    else:
        if not title:
            title = 'Zone Visualization'
    
    ax.set_title(title)
    
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
    if zones:
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
    
    return fig


def print_zone_summary(zones: List[BoundingBox]) -> None:
    """
    Print a summary of zone information.
    
    Args:
        zones: List of zones to summarize
    """
    print_info("Zone Summary:")
    total_area = 0
    
    for zone in zones:
        width = zone.x_max - zone.x_min
        height = zone.y_max - zone.y_min
        area = width * height
        total_area += area
        
        print_info(f"  {zone.name}:")
        print_info(f"    Coordinates: ({zone.x_min:.1f}, {zone.y_min:.1f}) to ({zone.x_max:.1f}, {zone.y_max:.1f})")
        print_info(f"    Dimensions: {width:.1f} × {height:.1f} mm")
        print_info(f"    Area: {area:.1f} mm²")
        if zone.description:
            print_info(f"    Description: {zone.description}")
    
    print_info(f"Total area covered: {total_area:.1f} mm²")


def quick_visualize(zones_file: Path, tracking_file: Optional[Path] = None) -> None:
    """
    Quick visualization function for zones.
    
    Args:
        zones_file: Path to zones JSON file
        tracking_file: Optional tracking data file
    """
    zones = load_zones_from_file(zones_file)
    print_zone_summary(zones)
    
    fig = visualize_zones(zones, tracking_file)
    plt.show()


if __name__ == "__main__":
    # Simple command-line interface for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python zone_visualizer.py <zones_file.json> [tracking_file.txt]")
        sys.exit(1)
    
    zones_file = Path(sys.argv[1])
    tracking_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    quick_visualize(zones_file, tracking_file)
