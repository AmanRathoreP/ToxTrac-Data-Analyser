"""
Simple GUI for interactive zone creation using matplotlib.

This module provides a lightweight GUI for users to visually define zones
by clicking on tracking data plots or empty coordinate spaces.

Author: Aman Rathore
Contact: amanr.me | amanrathore9753 <at> gmail <dot> com
Created on: Monday, July 14, 2025 at 10:40
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, TextBox
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from .models import BoundingBox, TrackingSession
from .data_loader import ToxTracDataLoader
from .logging_config import get_logger, print_success, print_error, print_info

logger = get_logger("zone_gui")


class ZoneCreatorGUI:
    """
    Interactive GUI for creating zones by clicking on plots.
    """
    
    def __init__(self):
        """Initialize the zone creator GUI."""
        self.logger = get_logger("zone_gui")
        self.zones = []
        self.current_zone_name = ""
        self.current_zone_points = []
        self.fig = None
        self.ax = None
        self.tracking_data = None
        self.background_image = None
        self.image_bounds = None
        
        # GUI state
        self.creating_zone = False
        self.zone_rectangles = []
        
    def create_zones_from_tracking_data(
        self, 
        tracking_file: Path, 
        output_file: Path
    ) -> None:
        """
        Create zones interactively using existing tracking data as background.
        
        Args:
            tracking_file: Path to tracking data file
            output_file: Path to save zone definitions
        """
        print_info("Loading tracking data for zone creation...")
        
        try:
            # Load tracking data
            loader = ToxTracDataLoader()
            session = loader.load_tracking_data(tracking_file)
            self.tracking_data = session
            
            # Create the GUI
            self._setup_gui_with_data(session)
            self._setup_controls()
            
            print_info("GUI opened! Use the interface to create zones:")
            print_info("1. Enter zone name in the text box")
            print_info("2. Click 'Start Zone' button")
            print_info("3. Click two points to define rectangle (top-left, bottom-right)")
            print_info("4. Repeat for more zones")
            print_info("5. Click 'Save Zones' when done")
            
            plt.show()
            
            # Save zones after GUI closes
            if self.zones:
                self._save_zones(output_file)
                
        except Exception as e:
            self.logger.error(f"Failed to create GUI: {e}")
            print_error(f"Failed to create GUI: {e}")
    
    def create_zones_from_scratch(
        self, 
        output_file: Path,
        x_range: Tuple[float, float] = (-200, 200),
        y_range: Tuple[float, float] = (-200, 200)
    ) -> None:
        """
        Create zones on an empty coordinate system.
        
        Args:
            output_file: Path to save zone definitions
            x_range: X-axis range for the plot
            y_range: Y-axis range for the plot
        """
        print_info("Creating zones on empty coordinate system...")
        
        try:
            # Create empty plot
            self._setup_empty_gui(x_range, y_range)
            self._setup_controls()
            
            print_info("GUI opened! Use the interface to create zones:")
            print_info("1. Enter zone name in the text box")
            print_info("2. Click 'Start Zone' button")
            print_info("3. Click two points to define rectangle (top-left, bottom-right)")
            print_info("4. Repeat for more zones")
            print_info("5. Click 'Save Zones' when done")
            
            plt.show()
            
            # Save zones after GUI closes
            if self.zones:
                self._save_zones(output_file)
                
        except Exception as e:
            self.logger.error(f"Failed to create GUI: {e}")
            print_error(f"Failed to create GUI: {e}")

    def create_zones_with_background_image(
        self,
        image_file: Path,
        image_bounds: Tuple[float, float, float, float],
        output_file: Path
    ) -> None:
        """
        Create zones interactively using a background image for reference.
        
        Args:
            image_file: Path to background image file
            image_bounds: Image coordinate bounds as (x_min, y_min, x_max, y_max)
            output_file: Path to save zone definitions
        """
        print_info(f"Loading background image for zone creation: {image_file}")
        
        try:
            # Load and store image data
            import matplotlib.image as mpimg
            self.background_image = mpimg.imread(image_file)
            self.image_bounds = image_bounds
            
            # Create the GUI
            self._setup_gui_with_image(image_file, image_bounds)
            self._setup_controls()
            
            print_info("GUI opened! Use the interface to create zones:")
            print_info("1. Enter zone name in the text box")
            print_info("2. Click 'Start Zone' button")
            print_info("3. Click two points to define rectangle (top-left, bottom-right)")
            print_info("4. Repeat for more zones")
            print_info("5. Click 'Save Zones' when done")
            
            plt.show()
            
            # Save zones after GUI closes
            if self.zones:
                self._save_zones(output_file)
                
        except Exception as e:
            self.logger.error(f"Failed to create GUI with image: {e}")
            print_error(f"Failed to create GUI with image: {e}")
    
    def _setup_gui_with_data(self, session: TrackingSession) -> None:
        """Setup GUI with tracking data as background."""
        # Create figure with subplots for controls
        self.fig = plt.figure(figsize=(12, 8))
        
        # Main plot area
        self.ax = plt.subplot2grid((6, 4), (0, 0), colspan=4, rowspan=4)
        
        # Plot tracking data
        df = session.get_dataframe()
        self.ax.scatter(df['x'], df['y'], alpha=0.3, s=1, c='blue', label='Tracking data')
        
        self.ax.set_xlabel('X Coordinate (mm)')
        self.ax.set_ylabel('Y Coordinate (mm)')
        self.ax.set_title(f'Zone Creator - {session.session_name}')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Set equal aspect ratio
        self.ax.set_aspect('equal')
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
    
    def _setup_empty_gui(self, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> None:
        """Setup GUI with empty coordinate system."""
        # Create figure with subplots for controls
        self.fig = plt.figure(figsize=(12, 8))
        
        # Main plot area
        self.ax = plt.subplot2grid((6, 4), (0, 0), colspan=4, rowspan=4)
        
        self.ax.set_xlim(x_range)
        self.ax.set_ylim(y_range)
        self.ax.set_xlabel('X Coordinate (mm)')
        self.ax.set_ylabel('Y Coordinate (mm)')
        self.ax.set_title('Zone Creator - Empty Coordinate System')
        self.ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio
        self.ax.set_aspect('equal')
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _setup_gui_with_image(self, image_file: Path, image_bounds: Tuple[float, float, float, float]) -> None:
        """Setup GUI with background image."""
        # Create figure with subplots for controls
        self.fig = plt.figure(figsize=(12, 8))
        
        # Main plot area
        self.ax = plt.subplot2grid((6, 4), (0, 0), colspan=4, rowspan=4)
        
        # Display background image
        x_min, y_min, x_max, y_max = image_bounds
        self.ax.imshow(self.background_image, extent=[x_min, x_max, y_min, y_max], 
                      aspect='auto', alpha=0.7, origin='lower')
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlabel('X Coordinate (mm)')
        self.ax.set_ylabel('Y Coordinate (mm)')
        self.ax.set_title(f'Zone Creator - Background Image: {image_file.name}')
        self.ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio
        self.ax.set_aspect('equal')
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
    
    def _setup_controls(self) -> None:
        """Setup control buttons and text boxes."""
        # Zone name input
        ax_name = plt.subplot2grid((6, 4), (4, 0), colspan=2)
        ax_name.text(0.1, 0.5, "Zone Name:", transform=ax_name.transAxes, fontsize=10)
        ax_name.axis('off')
        
        ax_textbox = plt.subplot2grid((6, 4), (4, 2), colspan=2)
        self.textbox = TextBox(ax_textbox, '', initial='zone_1')
        
        # Start Zone button
        ax_start = plt.subplot2grid((6, 4), (5, 0))
        self.btn_start = Button(ax_start, 'Start Zone', color='lightgreen')
        self.btn_start.on_clicked(self._start_zone)
        
        # Clear Last button
        ax_clear = plt.subplot2grid((6, 4), (5, 1))
        self.btn_clear = Button(ax_clear, 'Clear Last', color='orange')
        self.btn_clear.on_clicked(self._clear_last)
        
        # Save Zones button
        ax_save = plt.subplot2grid((6, 4), (5, 2))
        self.btn_save = Button(ax_save, 'Save Zones', color='lightblue')
        self.btn_save.on_clicked(self._save_and_close)
        
        # Quit button
        ax_quit = plt.subplot2grid((6, 4), (5, 3))
        self.btn_quit = Button(ax_quit, 'Quit', color='lightcoral')
        self.btn_quit.on_clicked(self._quit)
        
        # Status text
        self.status_text = self.fig.text(0.02, 0.02, "Ready. Enter zone name and click 'Start Zone'", 
                                        fontsize=10, color='blue')
        
        plt.tight_layout()
    
    def _start_zone(self, event) -> None:
        """Start creating a new zone."""
        zone_name = self.textbox.text.strip()
        if not zone_name:
            self._update_status("Please enter a zone name first!", 'red')
            return
        
        self.current_zone_name = zone_name
        self.current_zone_points = []
        self.creating_zone = True
        
        self._update_status(f"Creating zone '{zone_name}'. Click two points: top-left, then bottom-right", 'green')
    
    def _on_click(self, event) -> None:
        """Handle mouse clicks on the plot."""
        if not self.creating_zone or event.inaxes != self.ax:
            return
        
        # Add point to current zone
        self.current_zone_points.append((event.xdata, event.ydata))
        
        # Plot the point
        self.ax.plot(event.xdata, event.ydata, 'ro', markersize=8)
        self.fig.canvas.draw()
        
        if len(self.current_zone_points) == 1:
            self._update_status(f"First point selected. Click bottom-right corner.", 'green')
        elif len(self.current_zone_points) == 2:
            # Create the zone
            self._complete_zone()
    
    def _complete_zone(self) -> None:
        """Complete the current zone creation."""
        if len(self.current_zone_points) != 2:
            return
        
        # Get points
        p1 = self.current_zone_points[0]
        p2 = self.current_zone_points[1]
        
        # Calculate bounding box
        x_min = min(p1[0], p2[0])
        x_max = max(p1[0], p2[0])
        y_min = min(p1[1], p2[1])
        y_max = max(p1[1], p2[1])
        
        # Create zone
        zone = BoundingBox(
            name=self.current_zone_name,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            description=f"Interactively created zone"
        )
        
        self.zones.append(zone)
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x_min, y_min), 
            x_max - x_min, 
            y_max - y_min,
            linewidth=2, 
            edgecolor='red', 
            facecolor='red', 
            alpha=0.3,
            label=self.current_zone_name
        )
        self.ax.add_patch(rect)
        self.zone_rectangles.append(rect)
        
        # Add zone label
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        self.ax.text(center_x, center_y, self.current_zone_name, 
                    ha='center', va='center', fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.fig.canvas.draw()
        
        # Reset state
        self.creating_zone = False
        self.current_zone_points = []
        
        # Update zone name for next zone
        try:
            # Try to increment number in zone name
            if self.current_zone_name.endswith(tuple('0123456789')):
                base = self.current_zone_name.rstrip('0123456789')
                num = int(self.current_zone_name[len(base):]) + 1
                self.textbox.set_val(f"{base}{num}")
            else:
                self.textbox.set_val(f"{self.current_zone_name}_2")
        except:
            self.textbox.set_val(f"zone_{len(self.zones) + 1}")
        
        self._update_status(f"Zone '{self.current_zone_name}' created! Total zones: {len(self.zones)}", 'blue')
    
    def _clear_last(self, event) -> None:
        """Clear the last created zone."""
        if self.zones:
            # Remove last zone
            removed_zone = self.zones.pop()
            
            # Remove rectangle from plot
            if self.zone_rectangles:
                rect = self.zone_rectangles.pop()
                rect.remove()
                
                # Remove any text annotations (simplified - removes all text)
                for txt in self.ax.texts:
                    if txt.get_text() == removed_zone.name:
                        txt.remove()
                        break
                
                self.fig.canvas.draw()
            
            self._update_status(f"Removed zone '{removed_zone.name}'. Total zones: {len(self.zones)}", 'orange')
        else:
            self._update_status("No zones to remove!", 'red')
    
    def _save_and_close(self, event) -> None:
        """Save zones and close GUI."""
        if not self.zones:
            self._update_status("No zones to save!", 'red')
            return
        
        self._update_status(f"Saving {len(self.zones)} zones...", 'blue')
        plt.close(self.fig)
    
    def _quit(self, event) -> None:
        """Quit without saving."""
        plt.close(self.fig)
    
    def _update_status(self, message: str, color: str = 'blue') -> None:
        """Update status message."""
        self.status_text.set_text(message)
        self.status_text.set_color(color)
        self.fig.canvas.draw()
    
    def _save_zones(self, output_file: Path) -> None:
        """Save zones to JSON file."""
        try:
            zones_data = []
            for zone in self.zones:
                zones_data.append({
                    'name': zone.name,
                    'x_min': zone.x_min,
                    'y_min': zone.y_min,
                    'x_max': zone.x_max,
                    'y_max': zone.y_max,
                    'description': zone.description
                })
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(zones_data, f, indent=2)
            
            print_success(f"Saved {len(self.zones)} zones to {output_file}")
            
            # Print summary
            print_info("Zone Summary:")
            for zone in self.zones:
                print_info(f"  {zone.name}: ({zone.x_min:.1f}, {zone.y_min:.1f}) to ({zone.x_max:.1f}, {zone.y_max:.1f})")
                
        except Exception as e:
            self.logger.error(f"Failed to save zones: {e}")
            print_error(f"Failed to save zones: {e}")


def create_zones_gui_from_data(tracking_file: Path, output_file: Path) -> None:
    """
    Convenience function to create zones GUI with tracking data.
    
    Args:
        tracking_file: Path to tracking data file
        output_file: Path to save zone definitions
    """
    gui = ZoneCreatorGUI()
    gui.create_zones_from_tracking_data(tracking_file, output_file)


def create_zones_gui_empty(
    output_file: Path, 
    x_range: Tuple[float, float] = (-200, 200),
    y_range: Tuple[float, float] = (-200, 200)
) -> None:
    """
    Convenience function to create zones GUI with empty coordinate system.
    
    Args:
        output_file: Path to save zone definitions
        x_range: X-axis range for the plot
        y_range: Y-axis range for the plot
    """
    gui = ZoneCreatorGUI()
    gui.create_zones_gui_from_scratch(output_file, x_range, y_range)


def create_zones_gui_with_image(
    image_file: Path,
    image_bounds: Tuple[float, float, float, float],
    output_file: Path
) -> None:
    """
    Convenience function to create zones GUI with background image.
    
    Args:
        image_file: Path to background image file
        image_bounds: Image coordinate bounds as (x_min, y_min, x_max, y_max)
        output_file: Path to save zone definitions
    """
    gui = ZoneCreatorGUI()
    gui.create_zones_with_background_image(image_file, image_bounds, output_file)
