"""
Analysis engine for ToxTrac tracking data.

This module contains the core analysis algorithms for calculating
time spent in zones, movement patterns, and other behavioral metrics.

Author: Aman Rathore
Contact: amanr.me | amanrathore9753 <at> gmail <dot> com
Created on: Monday, July 14, 2025 at 10:40
"""

from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from .models import (
    TrackingSession, BoundingBox, ZoneAnalysisResult, 
    SessionAnalysisResult, TrackingPoint
)
from .logging_config import get_logger, log_function_call

logger = get_logger("analyzer")


class ToxTracAnalyzer:
    """
    Main analysis engine for ToxTrac tracking data.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.logger = get_logger("analyzer")
    
    @log_function_call(logger)
    def analyze_zone_occupancy(
        self, 
        session: TrackingSession, 
        zone: BoundingBox
    ) -> ZoneAnalysisResult:
        """
        Analyze time spent in a specific zone.
        
        Args:
            session: Tracking session to analyze
            zone: Zone/bounding box to analyze
            
        Returns:
            ZoneAnalysisResult with detailed occupancy statistics
        """
        self.logger.info(f"Analyzing zone '{zone.name}' for session '{session.session_name}'")
        
        # Convert to DataFrame for easier processing
        df = session.get_dataframe()
        
        if len(df) < 2:
            # Return empty result for insufficient data
            return ZoneAnalysisResult(
                zone_name=zone.name,
                total_time_inside=0.0,
                total_time_outside=0.0,
                percentage_inside=0.0,
                entry_count=0,
                exit_count=0,
                mean_visit_duration=0.0,
                longest_visit=0.0,
                shortest_visit=0.0,
                distance_inside=0.0,
                distance_outside=0.0,
                mean_speed_inside=0.0,
                mean_speed_outside=0.0
            )
        
        # Check which points are inside the zone
        df['in_zone'] = df.apply(
            lambda row: zone.contains_point(row['x'], row['y']), 
            axis=1
        )
        
        # Calculate movement statistics with gap awareness
        from .utils import calculate_distance_with_gaps
        
        distances = calculate_distance_with_gaps(df)
        
        df['distance'] = distances
        df['dt'] = df['time'].diff()
        df['speed'] = (df['distance'] / df['dt']).fillna(0)
        
        # Log gap information if gaps were detected
        from .utils import get_gap_statistics
        gap_info = get_gap_statistics(df)
        if gap_info['total_gaps'] > 0:
            self.logger.info(
                f"Detected {gap_info['total_gaps']} gaps in tracking data, "
                f"total gap duration: {gap_info['total_gap_duration']:.2f}s"
            )
        
        # Calculate time spent inside and outside
        total_time = session.total_duration
        
        # Group consecutive time periods in/out of zone
        df['zone_change'] = df['in_zone'] != df['in_zone'].shift(1)
        df['visit_id'] = df['zone_change'].cumsum()
        
        # Calculate visit statistics
        visits = df.groupby('visit_id').agg({
            'in_zone': 'first',
            'time': ['min', 'max', 'count']
        }).reset_index()
        
        visits.columns = ['visit_id', 'in_zone', 'start_time', 'end_time', 'point_count']
        visits['duration'] = visits['end_time'] - visits['start_time']
        
        # Filter for inside visits only
        inside_visits = visits[visits['in_zone'] == True]
        outside_visits = visits[visits['in_zone'] == False]
        
        # Calculate time statistics
        total_time_inside = inside_visits['duration'].sum() if len(inside_visits) > 0 else 0
        total_time_outside = outside_visits['duration'].sum() if len(outside_visits) > 0 else total_time
        
        percentage_inside = (total_time_inside / total_time * 100) if total_time > 0 else 0
        
        entry_count = len(inside_visits)
        exit_count = len(inside_visits)  # Same as entry count for complete visits
        
        if len(inside_visits) > 0:
            mean_visit_duration = inside_visits['duration'].mean()
            longest_visit = inside_visits['duration'].max()
            shortest_visit = inside_visits['duration'].min()
        else:
            mean_visit_duration = 0
            longest_visit = 0
            shortest_visit = 0
        
        # Calculate distance and speed statistics per zone
        inside_points = df[df['in_zone'] == True]
        outside_points = df[df['in_zone'] == False]
        
        distance_inside = inside_points['distance'].sum() if len(inside_points) > 0 else 0.0
        distance_outside = outside_points['distance'].sum() if len(outside_points) > 0 else 0.0
        
        mean_speed_inside = inside_points['speed'].mean() if len(inside_points) > 0 else 0.0
        mean_speed_outside = outside_points['speed'].mean() if len(outside_points) > 0 else 0.0
        
        # Handle NaN values
        mean_speed_inside = 0.0 if pd.isna(mean_speed_inside) else mean_speed_inside
        mean_speed_outside = 0.0 if pd.isna(mean_speed_outside) else mean_speed_outside
        
        result = ZoneAnalysisResult(
            zone_name=zone.name,
            total_time_inside=total_time_inside,
            total_time_outside=total_time_outside,
            percentage_inside=percentage_inside,
            entry_count=entry_count,
            exit_count=exit_count,
            mean_visit_duration=mean_visit_duration,
            longest_visit=longest_visit,
            shortest_visit=shortest_visit,
            distance_inside=distance_inside,
            distance_outside=distance_outside,
            mean_speed_inside=mean_speed_inside,
            mean_speed_outside=mean_speed_outside
        )
        
        self.logger.info(
            f"Zone '{zone.name}': {percentage_inside:.1f}% occupancy, "
            f"{entry_count} entries, {mean_visit_duration:.2f}s avg visit, "
            f"{distance_inside:.1f}mm distance inside"
        )
        
        return result
    
    @log_function_call(logger)
    def analyze_session(
        self,
        session: TrackingSession,
        zones: List[BoundingBox],
        calculate_movement: bool = True
    ) -> SessionAnalysisResult:
        """
        Perform complete analysis of a tracking session.
        
        Args:
            session: Tracking session to analyze
            zones: List of zones to analyze
            calculate_movement: Whether to calculate movement statistics
            
        Returns:
            SessionAnalysisResult with complete analysis
        """
        self.logger.info(f"Analyzing session '{session.session_name}' with {len(zones)} zones")
        
        # Analyze each zone
        zone_results = []
        for zone in tqdm(zones, desc=f"Analyzing zones for {session.session_name}"):
            result = self.analyze_zone_occupancy(session, zone)
            zone_results.append(result)
        
        # Calculate movement statistics if requested
        total_distance = None
        mean_speed = None
        
        if calculate_movement:
            total_distance, mean_speed = self._calculate_movement_stats(session)
        
        result = SessionAnalysisResult(
            session_name=session.session_name,
            zone_results=zone_results,
            total_duration=session.total_duration,
            total_distance=total_distance,
            mean_speed=mean_speed
        )
        
        self.logger.info(f"Completed analysis for session '{session.session_name}'")
        return result
    
    @log_function_call(logger)
    def analyze_multiple_sessions(
        self,
        sessions: List[TrackingSession],
        zones: List[BoundingBox],
        calculate_movement: bool = True
    ) -> List[SessionAnalysisResult]:
        """
        Analyze multiple tracking sessions.
        
        Args:
            sessions: List of tracking sessions to analyze
            zones: List of zones to analyze for each session
            calculate_movement: Whether to calculate movement statistics
            
        Returns:
            List of SessionAnalysisResult objects
        """
        self.logger.info(f"Analyzing {len(sessions)} sessions")
        
        results = []
        for session in tqdm(sessions, desc="Analyzing sessions"):
            try:
                result = self.analyze_session(session, zones, calculate_movement)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error analyzing session '{session.session_name}': {e}")
                continue
        
        self.logger.info(f"Successfully analyzed {len(results)} out of {len(sessions)} sessions")
        return results
    
    def _calculate_movement_stats(self, session: TrackingSession) -> Tuple[float, float]:
        """
        Calculate movement statistics for a session.
        
        Args:
            session: Tracking session to analyze
            
        Returns:
            Tuple of (total_distance, mean_speed)
        """
        df = session.get_dataframe()
        
        if len(df) < 2:
            return 0.0, 0.0
        
        # Calculate distances with gap awareness
        from .utils import calculate_distance_with_gaps, get_gap_statistics
        
        distances = calculate_distance_with_gaps(df)
        df['distance'] = distances
        
        # Get gap statistics separately
        gap_info = get_gap_statistics(df)
        df['dt'] = df['time'].diff()
        df['speed'] = (df['distance'] / df['dt']).fillna(0)
        
        # Log gap information for movement stats if gaps detected
        if gap_info['total_gaps'] > 0:
            self.logger.debug(
                f"Movement stats: Found {gap_info['total_gaps']} gaps, "
                f"skipped {gap_info['total_gap_duration']:.2f}s of missing data"
            )
        
        # Remove NaN values and zeros from calculation
        valid_data = df[df['distance'] > 0]
        
        total_distance = df['distance'].sum()
        mean_speed = valid_data['speed'].mean() if len(valid_data) > 0 else 0.0
        
        return total_distance, mean_speed
    
    def create_trajectory_plot(
        self,
        session: TrackingSession,
        zones: List[BoundingBox],
        output_file: Optional[Path] = None,
        show_zones: bool = True,
        color_by_zone: bool = True,
        show_speed: bool = False,
        figsize: tuple = (12, 10)
    ):
        """
        Create a trajectory plot showing the animal's path with zones.
        
        Args:
            session: Tracking session to plot
            zones: List of zones to overlay
            output_file: Optional path to save the plot
            show_zones: Whether to show zone boundaries
            color_by_zone: Whether to color trajectory by current zone
            show_speed: Whether to color trajectory by speed
            figsize: Figure size as (width, height)
            
        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib.colors as mcolors
        
        self.logger.info(f"Creating trajectory plot for session '{session.session_name}'")
        
        # Get tracking data
        df = session.get_dataframe()
        
        if len(df) < 2:
            self.logger.warning("Insufficient data for trajectory plot")
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.text(0.5, 0.5, "Insufficient data for trajectory plot", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Calculate movement statistics
        df['dx'] = df['x'].diff()
        df['dy'] = df['y'].diff()
        df['dt'] = df['time'].diff()
        df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2).fillna(0)
        df['speed'] = (df['distance'] / df['dt']).fillna(0)
        
        # Determine which zone each point is in
        if color_by_zone and zones:
            df['current_zone'] = 'none'
            for zone in zones:
                mask = df.apply(lambda row: zone.contains_point(row['x'], row['y']), axis=1)
                df.loc[mask, 'current_zone'] = zone.name
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot trajectory
        if show_speed and not color_by_zone:
            # Color by speed
            speeds = df['speed'].values
            scatter = ax.scatter(df['x'], df['y'], c=speeds, s=2, alpha=0.7, 
                               cmap='viridis', label='Trajectory')
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Speed (mm/s)')
            
        elif color_by_zone and zones:
            # Color by current zone
            zone_colors = {}
            colors = list(mcolors.TABLEAU_COLORS.values())
            
            for i, zone in enumerate(zones):
                zone_colors[zone.name] = colors[i % len(colors)]
            zone_colors['none'] = 'lightgray'
            
            for zone_name in df['current_zone'].unique():
                mask = df['current_zone'] == zone_name
                if mask.any():
                    color = zone_colors.get(zone_name, 'lightgray')
                    ax.scatter(df.loc[mask, 'x'], df.loc[mask, 'y'], 
                             c=color, s=2, alpha=0.7, label=f'In {zone_name}')
            
        else:
            # Simple trajectory
            ax.plot(df['x'], df['y'], 'b-', alpha=0.7, linewidth=1, label='Trajectory')
            ax.scatter(df['x'].iloc[0], df['y'].iloc[0], c='green', s=50, 
                      marker='o', label='Start', zorder=5)
            ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], c='red', s=50, 
                      marker='s', label='End', zorder=5)
        
        # Plot zones
        if show_zones:
            colors = list(mcolors.TABLEAU_COLORS.values())
            for i, zone in enumerate(zones):
                color = colors[i % len(colors)] if color_by_zone else 'red'
                
                width = zone.x_max - zone.x_min
                height = zone.y_max - zone.y_min
                
                rect = patches.Rectangle(
                    (zone.x_min, zone.y_min), 
                    width, 
                    height,
                    linewidth=2, 
                    edgecolor=color, 
                    facecolor='none',
                    linestyle='--',
                    alpha=0.8,
                    zorder=3
                )
                ax.add_patch(rect)
                
                # Add zone label
                center_x = zone.x_min + width / 2
                center_y = zone.y_min + height / 2
                ax.text(center_x, center_y, zone.name, 
                       ha='center', va='center', fontweight='bold', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       zorder=4)
        
        # Set axis properties
        ax.set_xlabel('X Coordinate (mm)')
        ax.set_ylabel('Y Coordinate (mm)')
        ax.set_title(f'Trajectory Plot - {session.session_name}')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')
        
        # Add statistics text
        total_distance = df['distance'].sum()
        mean_speed = df['speed'].mean()
        max_speed = df['speed'].max()
        
        stats_text = (
            f"Duration: {session.total_duration:.1f}s\n"
            f"Total Distance: {total_distance:.1f}mm\n"
            f"Mean Speed: {mean_speed:.1f}mm/s\n"
            f"Max Speed: {max_speed:.1f}mm/s"
        )
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Trajectory plot saved to: {output_file}")
        
        return fig
    
    def create_zone_occupancy_plot(
        self,
        result: SessionAnalysisResult,
        output_file: Optional[Path] = None,
        figsize: tuple = (10, 6)
    ):
        """
        Create a bar chart showing time spent in each zone.
        
        Args:
            result: Session analysis result
            output_file: Optional path to save the plot
            figsize: Figure size as (width, height)
            
        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Extract zone data
        zone_names = [zone_result.zone_name for zone_result in result.zone_results]
        percentages = [zone_result.percentage_inside for zone_result in result.zone_results]
        times = [zone_result.total_time_inside for zone_result in result.zone_results]
        entries = [zone_result.entries for zone_result in result.zone_results]
        
        # Colors for zones
        colors = plt.cm.Set3(np.linspace(0, 1, len(zone_names)))
        
        # Plot 1: Time percentage
        bars1 = ax1.bar(zone_names, percentages, color=colors)
        ax1.set_ylabel('Time Spent (%)')
        ax1.set_title('Time Spent in Each Zone')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, pct in zip(bars1, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        # Plot 2: Entry counts
        bars2 = ax2.bar(zone_names, entries, color=colors)
        ax2.set_ylabel('Number of Entries')
        ax2.set_title('Zone Entry Frequency')
        
        # Add value labels on bars
        for bar, entry in zip(bars2, entries):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{entry}', ha='center', va='bottom')
        
        # Rotate x-axis labels if needed
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Zone Occupancy Analysis - {result.session_name}')
        plt.tight_layout()
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Zone occupancy plot saved to {output_file}")
        
        return fig
    
    def create_speed_distribution_plot(
        self,
        session: TrackingSession,
        output_file: Optional[Path] = None,
        figsize: tuple = (12, 8)
    ):
        """
        Create speed distribution histogram and time series.
        
        Args:
            session: Tracking session to analyze
            output_file: Optional path to save the plot
            figsize: Figure size as (width, height)
            
        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        df = session.get_dataframe()
        
        if len(df) < 2:
            self.logger.warning(f"Not enough data points for speed analysis in session {session.session_name}")
            return None
        
        # Calculate speed
        df['dx'] = df['x'].diff()
        df['dy'] = df['y'].diff()
        df['dt'] = df['time'].diff()
        df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2).fillna(0)
        df['speed'] = (df['distance'] / df['dt']).fillna(0)
        
        # Remove outliers (speeds > 99th percentile)
        speed_99th = df['speed'].quantile(0.99)
        df_clean = df[df['speed'] <= speed_99th]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Speed histogram
        ax1.hist(df_clean['speed'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Speed (mm/s)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Speed Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_speed = df_clean['speed'].mean()
        median_speed = df_clean['speed'].median()
        max_speed = df_clean['speed'].max()
        
        ax1.axvline(mean_speed, color='red', linestyle='--', label=f'Mean: {mean_speed:.1f}')
        ax1.axvline(median_speed, color='orange', linestyle='--', label=f'Median: {median_speed:.1f}')
        ax1.legend()
        
        # Plot 2: Speed over time
        ax2.plot(df_clean['time'], df_clean['speed'], alpha=0.7, linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Speed (mm/s)')
        ax2.set_title('Speed Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Acceleration
        df_clean['acceleration'] = df_clean['speed'].diff() / df_clean['dt']
        df_clean['acceleration'] = df_clean['acceleration'].fillna(0)
        
        # Remove extreme acceleration outliers
        acc_99th = df_clean['acceleration'].abs().quantile(0.99)
        df_acc_clean = df_clean[df_clean['acceleration'].abs() <= acc_99th]
        
        ax3.hist(df_acc_clean['acceleration'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.set_xlabel('Acceleration (mm/s²)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Acceleration Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Distance vs Speed scatter
        ax4.scatter(df_clean['distance'], df_clean['speed'], alpha=0.3, s=1)
        ax4.set_xlabel('Step Distance (mm)')
        ax4.set_ylabel('Speed (mm/s)')
        ax4.set_title('Distance vs Speed')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Movement Analysis - {session.session_name}')
        plt.tight_layout()
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Speed distribution plot saved to {output_file}")
        
        return fig
    
    def create_movement_heatmap(
        self,
        session: TrackingSession,
        zones: List[BoundingBox],
        output_file: Optional[Path] = None,
        figsize: tuple = (10, 8),
        bins: int = 50
    ):
        """
        Create a movement density heatmap.
        
        Args:
            session: Tracking session to analyze
            zones: List of zones to overlay
            output_file: Optional path to save the plot
            figsize: Figure size as (width, height)
            bins: Number of bins for heatmap
            
        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        df = session.get_dataframe()
        
        if len(df) < 10:
            self.logger.warning(f"Not enough data points for heatmap in session {session.session_name}")
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            df['x'], df['y'], bins=bins, density=True
        )
        
        # Plot heatmap
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Density (relative frequency)')
        
        # Overlay zones
        colors = ['cyan', 'lime', 'yellow', 'magenta', 'white']
        for i, zone in enumerate(zones):
            color = colors[i % len(colors)]
            
            # Draw zone boundary
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (zone.x_min, zone.y_min),
                zone.x_max - zone.x_min,
                zone.y_max - zone.y_min,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                label=zone.name
            )
            ax.add_patch(rect)
        
        ax.set_xlabel('X Coordinate (mm)')
        ax.set_ylabel('Y Coordinate (mm)')
        ax.set_title(f'Movement Density Heatmap - {session.session_name}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Movement heatmap saved to {output_file}")
        
        return fig
    
    def create_timeseries_plot(
        self,
        session: TrackingSession,
        zones: List[BoundingBox],
        output_file: Optional[Path] = None,
        figsize: tuple = (12, 8),
        window_size: int = 100
    ):
        """
        Create time series analysis showing zone occupancy over time.
        
        Args:
            session: Tracking session to analyze
            zones: List of zones to analyze
            output_file: Optional path to save the plot
            figsize: Figure size as (width, height)
            window_size: Moving average window size
            
        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        
        df = session.get_dataframe()
        
        if len(df) < window_size * 2:
            self.logger.warning(f"Not enough data points for time series analysis in session {session.session_name}")
            return None
        
        # Determine which zone each point is in
        for zone in zones:
            zone_mask = (
                (df['x'] >= zone.x_min) & (df['x'] <= zone.x_max) &
                (df['y'] >= zone.y_min) & (df['y'] <= zone.y_max)
            )
            df[f'in_{zone.name}'] = zone_mask.astype(int)
        
        # Calculate speed
        df['dx'] = df['x'].diff()
        df['dy'] = df['y'].diff()
        df['dt'] = df['time'].diff()
        df['speed'] = (np.sqrt(df['dx']**2 + df['dy']**2) / df['dt']).fillna(0)
        
        fig, axes = plt.subplots(len(zones) + 2, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Speed over time
        axes[0].plot(df['time'], df['speed'], alpha=0.7, linewidth=0.5)
        axes[0].set_ylabel('Speed (mm/s)')
        axes[0].set_title('Speed Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Add moving average
        if len(df) >= window_size:
            speed_ma = df['speed'].rolling(window=window_size, center=True).mean()
            axes[0].plot(df['time'], speed_ma, color='red', linewidth=2, label=f'MA({window_size})')
            axes[0].legend()
        
        # Plot 2: Distance from center over time
        center_x = df['x'].mean()
        center_y = df['y'].mean()
        df['distance_from_center'] = np.sqrt((df['x'] - center_x)**2 + (df['y'] - center_y)**2)
        
        axes[1].plot(df['time'], df['distance_from_center'], alpha=0.7, linewidth=0.5)
        axes[1].set_ylabel('Distance from Center (mm)')
        axes[1].set_title('Distance from Center Over Time')
        axes[1].grid(True, alpha=0.3)
        
        if len(df) >= window_size:
            dist_ma = df['distance_from_center'].rolling(window=window_size, center=True).mean()
            axes[1].plot(df['time'], dist_ma, color='red', linewidth=2, label=f'MA({window_size})')
            axes[1].legend()
        
        # Plots 3+: Zone occupancy over time
        colors = plt.cm.Set1(np.linspace(0, 1, len(zones)))
        
        for i, zone in enumerate(zones):
            ax = axes[i + 2]
            zone_col = f'in_{zone.name}'
            
            # Plot zone occupancy as filled area
            ax.fill_between(df['time'], 0, df[zone_col], alpha=0.5, color=colors[i], label=zone.name)
            
            # Add moving average of zone occupancy
            if len(df) >= window_size:
                zone_ma = df[zone_col].rolling(window=window_size, center=True).mean()
                ax.plot(df['time'], zone_ma, color=colors[i], linewidth=2)
            
            ax.set_ylabel(f'{zone.name} Occupancy')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        axes[-1].set_xlabel('Time (s)')
        plt.suptitle(f'Time Series Analysis - {session.session_name}')
        plt.tight_layout()
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Time series plot saved to {output_file}")
        
        return fig
    
    def create_session_comparison_plot(
        self,
        results: List[SessionAnalysisResult],
        output_file: Optional[Path] = None,
        figsize: tuple = (12, 8)
    ):
        """
        Create comparison plots across multiple sessions.
        
        Args:
            results: List of session analysis results
            output_file: Optional path to save the plot
            figsize: Figure size as (width, height)
            
        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if len(results) < 2:
            self.logger.warning("Need at least 2 sessions for comparison plot")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        session_names = [result.session_name for result in results]
        
        # Get all unique zone names
        all_zones = set()
        for result in results:
            all_zones.update([zone_result.zone_name for zone_result in result.zone_results])
        all_zones = sorted(list(all_zones))
        
        # Plot 1: Zone occupancy comparison
        x = np.arange(len(session_names))
        width = 0.8 / len(all_zones)
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_zones)))
        
        for i, zone in enumerate(all_zones):
            percentages = []
            for result in results:
                # Find this zone in the result
                zone_pct = 0
                for zone_result in result.zone_results:
                    if zone_result.zone_name == zone:
                        zone_pct = zone_result.percentage_inside
                        break
                percentages.append(zone_pct)
            
            ax1.bar(x + i * width, percentages, width, label=zone, color=colors[i])
        
        ax1.set_xlabel('Session')
        ax1.set_ylabel('Time Spent (%)')
        ax1.set_title('Zone Occupancy Comparison')
        ax1.set_xticks(x + width * (len(all_zones) - 1) / 2)
        ax1.set_xticklabels(session_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Total distance comparison
        distances = [result.total_distance or 0 for result in results]
        bars2 = ax2.bar(session_names, distances, color='skyblue')
        ax2.set_xlabel('Session')
        ax2.set_ylabel('Total Distance (mm)')
        ax2.set_title('Total Distance Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, dist in zip(bars2, distances):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(distances) * 0.01,
                    f'{dist:.0f}', ha='center', va='bottom')
        
        # Plot 3: Mean speed comparison
        speeds = [result.mean_speed or 0 for result in results]
        bars3 = ax3.bar(session_names, speeds, color='lightcoral')
        ax3.set_xlabel('Session')
        ax3.set_ylabel('Mean Speed (mm/s)')
        ax3.set_title('Mean Speed Comparison')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, speed in zip(bars3, speeds):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(speeds) * 0.01,
                    f'{speed:.1f}', ha='center', va='bottom')
        
        # Plot 4: Session duration comparison
        durations = [result.total_duration for result in results]
        bars4 = ax4.bar(session_names, durations, color='lightgreen')
        ax4.set_xlabel('Session')
        ax4.set_ylabel('Duration (s)')
        ax4.set_title('Session Duration Comparison')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, dur in zip(bars4, durations):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(durations) * 0.01,
                    f'{dur:.0f}', ha='center', va='bottom')
        
        plt.suptitle('Session Comparison Analysis')
        plt.tight_layout()
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Session comparison plot saved to {output_file}")
        
        return fig
    
    def create_zone_usage_summary_plot(
        self,
        results: List[SessionAnalysisResult],
        output_file: Optional[Path] = None,
        figsize: tuple = (12, 6)
    ):
        """
        Create summary plot of zone usage across all sessions.
        
        Args:
            results: List of session analysis results
            output_file: Optional path to save the plot
            figsize: Figure size as (width, height)
            
        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if len(results) < 1:
            self.logger.warning("Need at least 1 session for zone usage summary")
            return None
        
        # Collect zone data across all sessions
        zone_data = {}
        for result in results:
            for zone_result in result.zone_results:
                zone_name = zone_result.zone_name
                if zone_name not in zone_data:
                    zone_data[zone_name] = {
                        'percentages': [],
                        'entries': [],
                        'avg_visits': []
                    }
                
                zone_data[zone_name]['percentages'].append(zone_result.percentage_inside)
                zone_data[zone_name]['entries'].append(zone_result.entries)
                if zone_result.entries > 0:
                    avg_visit = zone_result.total_time_inside / zone_result.entries
                else:
                    avg_visit = 0
                zone_data[zone_name]['avg_visits'].append(avg_visit)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        zone_names = list(zone_data.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(zone_names)))
        
        # Plot 1: Average time spent with error bars
        means = [np.mean(zone_data[zone]['percentages']) for zone in zone_names]
        stds = [np.std(zone_data[zone]['percentages']) for zone in zone_names]
        
        bars1 = ax1.bar(zone_names, means, yerr=stds, capsize=5, color=colors)
        ax1.set_ylabel('Time Spent (%) - Mean ± SD')
        ax1.set_title('Average Zone Occupancy Across Sessions')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars1, means, stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Entry frequency
        entry_means = [np.mean(zone_data[zone]['entries']) for zone in zone_names]
        entry_stds = [np.std(zone_data[zone]['entries']) for zone in zone_names]
        
        bars2 = ax2.bar(zone_names, entry_means, yerr=entry_stds, capsize=5, color=colors)
        ax2.set_ylabel('Entries - Mean ± SD')
        ax2.set_title('Average Zone Entries Across Sessions')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars2, entry_means, entry_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Rotate x-axis labels if needed
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Zone Usage Summary (n={len(results)} sessions)')
        plt.tight_layout()
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Zone usage summary plot saved to {output_file}")
        
        return fig
    
    @log_function_call(logger)
    def create_summary_statistics(
        self, 
        results: List[SessionAnalysisResult]
    ) -> Dict[str, Any]:
        """
        Create summary statistics across multiple sessions.
        
        Args:
            results: List of session analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}
        
        self.logger.info(f"Creating summary statistics for {len(results)} sessions")
        
        # Collect data for each zone across all sessions
        zone_names = set()
        for result in results:
            for zone_result in result.zone_results:
                zone_names.add(zone_result.zone_name)
        
        zone_summaries = {}
        for zone_name in zone_names:
            zone_data = []
            for result in results:
                for zone_result in result.zone_results:
                    if zone_result.zone_name == zone_name:
                        zone_data.append(zone_result)
                        break
            
            if zone_data:
                zone_summaries[zone_name] = {
                    'n_sessions': len(zone_data),
                    'mean_percentage_inside': np.mean([z.percentage_inside for z in zone_data]),
                    'std_percentage_inside': np.std([z.percentage_inside for z in zone_data]),
                    'mean_entry_count': np.mean([z.entry_count for z in zone_data]),
                    'mean_visit_duration': np.mean([z.mean_visit_duration for z in zone_data]),
                }
        
        # Overall session statistics
        session_durations = [r.total_duration for r in results]
        session_distances = [r.total_distance for r in results if r.total_distance is not None]
        session_speeds = [r.mean_speed for r in results if r.mean_speed is not None]
        
        summary = {
            'n_sessions': len(results),
            'session_statistics': {
                'mean_duration': np.mean(session_durations),
                'std_duration': np.std(session_durations),
                'mean_distance': np.mean(session_distances) if session_distances else None,
                'std_distance': np.std(session_distances) if session_distances else None,
                'mean_speed': np.mean(session_speeds) if session_speeds else None,
                'std_speed': np.std(session_speeds) if session_speeds else None,
            },
            'zone_statistics': zone_summaries
        }
        
        self.logger.info("Summary statistics created successfully")
        return summary
