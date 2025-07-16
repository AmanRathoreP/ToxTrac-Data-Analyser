"""
Data loading and parsing utilities for ToxTrac data files.

This module handles reading and parsing various ToxTrac output files
including tracking data, zone definitions, and statistics.

Author: Aman Rathore
Contact: amanr.me | amanrathore9753 <at> gmail <dot> com
Created on: Monday, July 14, 2025 at 10:40
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import re
import numpy as np

from .models import TrackingPoint, TrackingSession, BoundingBox
from .logging_config import get_logger, log_function_call
from . import config

logger = get_logger("data_loader")


class ToxTracDataLoader:
    """
    Handles loading and parsing ToxTrac data files.
    """
    
    def __init__(self):
        """Initialize the data loader."""
        self.logger = get_logger("data_loader")
    
    @log_function_call(logger)
    def find_tracking_files(self, directory: Path) -> List[Path]:
        """
        Find all tracking files in a directory.
        
        Args:
            directory: Directory to search for tracking files
            
        Returns:
            List of paths to tracking files
        """
        tracking_files = []
        
        # Look for various tracking file patterns
        patterns = [
            "**/Tracking_RealSpace*.txt",
            "**/Tracking_*.txt"
        ]
        
        for pattern in patterns:
            files = list(directory.glob(pattern))
            tracking_files.extend(files)
        
        # Remove duplicates and sort
        tracking_files = sorted(list(set(tracking_files)))
        
        self.logger.info(
            f"Found {len(tracking_files)} tracking files in {directory}"
        )
        
        return tracking_files
    
    @log_function_call(logger)
    def load_tracking_data(self, file_path: Path) -> TrackingSession:
        """
        Load tracking data from a ToxTrac tracking file.
        
        Args:
            file_path: Path to the tracking file
            
        Returns:
            TrackingSession object with loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Tracking file not found: {file_path}")
        
        self.logger.info(f"Loading tracking data from {file_path}")
        
        try:
            # Read the file with tab separation
            df = pd.read_csv(file_path, sep='\t')
            
            # Standardize column names (handle different formats)
            df.columns = df.columns.str.strip()
            column_mapping = {
                'Time (sec)': 'time',
                'Video Seq.': 'video_seq',
                'Arena': 'arena', 
                'Track': 'track',
                'Pos. X (mm)': 'x',
                'Pos. Y (mm)': 'y',
                'Label': 'label'
            }
            
            # Rename columns if they exist
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df.rename(columns={old_name: new_name}, inplace=True)
            
            # Validate required columns
            required_columns = ['time', 'arena', 'track', 'x', 'y', 'label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {missing_columns}. "
                    f"Available columns: {list(df.columns)}"
                )
            
            # Handle missing data and convert to TrackingPoint objects
            data_points = []
            missing_points = []
            
            for idx, row in df.iterrows():
                # Check for missing coordinates (NaN or empty)
                x_val = row['x']
                y_val = row['y']
                
                # Convert to float, handling missing values
                try:
                    x = float(x_val) if pd.notna(x_val) and str(x_val).strip() != '' else None
                    y = float(y_val) if pd.notna(y_val) and str(y_val).strip() != '' else None
                except (ValueError, TypeError):
                    x = None
                    y = None
                
                # If coordinates are missing, mark for interpolation
                if x is None or y is None:
                    missing_points.append(idx)
                    # Store with NaN for now
                    x = float('nan') if x is None else x
                    y = float('nan') if y is None else y
                
                point = TrackingPoint(
                    time=float(row['time']),
                    arena=int(row['arena']),
                    track=int(row['track']),
                    x=x,
                    y=y,
                    label=int(row['label']),
                    video_seq=int(row['video_seq']) if 'video_seq' in row else None
                )
                data_points.append(point)
            
            # Log missing data information
            if missing_points:
                self.logger.warning(f"Found {len(missing_points)} data points with missing coordinates")
                
                # Apply interpolation if enabled
                config_settings = config.DEFAULT_ANALYSIS_SETTINGS
                if config_settings.get('interpolate_missing_points', True):  # Default to True now
                    data_points = self._interpolate_missing_data(data_points, missing_points)
                else:
                    self.logger.warning("Interpolation disabled - keeping NaN values")
            
            # Calculate session statistics
            total_duration = data_points[-1].time - data_points[0].time if data_points else 0
            sampling_rate = len(data_points) / total_duration if total_duration > 0 else 0
            
            session = TrackingSession(
                session_name=file_path.stem,
                data_points=data_points,
                source_file=file_path,
                total_duration=total_duration,
                sampling_rate=sampling_rate
            )
            
            self.logger.info(
                f"Loaded {len(data_points)} data points, "
                f"duration: {total_duration:.2f}s, "
                f"sampling rate: {sampling_rate:.2f}Hz"
            )
            
            return session
            
        except Exception as e:
            self.logger.error(f"Error loading tracking data from {file_path}: {e}")
            raise
    
    def _interpolate_missing_data(
        self, 
        data_points: List[TrackingPoint], 
        missing_indices: List[int]
    ) -> List[TrackingPoint]:
        """
        Interpolate missing coordinates using linear interpolation.
        
        Args:
            data_points: List of tracking points with potential NaN values
            missing_indices: Indices of points with missing data
            
        Returns:
            List of tracking points with interpolated coordinates
        """
        if not missing_indices:
            return data_points
        
        # Convert to arrays for easier manipulation
        times = np.array([p.time for p in data_points])
        x_coords = np.array([p.x for p in data_points])
        y_coords = np.array([p.y for p in data_points])
        
        # Find consecutive missing segments
        missing_segments = []
        current_segment = []
        
        for idx in missing_indices:
            if not current_segment or idx == current_segment[-1] + 1:
                current_segment.append(idx)
            else:
                missing_segments.append(current_segment)
                current_segment = [idx]
        
        if current_segment:
            missing_segments.append(current_segment)
        
        # Interpolate each segment
        interpolated_count = 0
        for segment in missing_segments:
            segment_size = len(segment)
            interpolated_count += segment_size
            
            # Warn if interpolating many consecutive points
            if segment_size > 10:
                self.logger.warning(
                    f"Interpolating {segment_size} consecutive missing points "
                    f"(indices {segment[0]}-{segment[-1]}). "
                    f"This may indicate a significant gap in tracking data."
                )
            
            start_idx = segment[0]
            end_idx = segment[-1]
            
            # Find valid points before and after the segment
            before_idx = None
            after_idx = None
            
            # Look for valid point before
            for i in range(start_idx - 1, -1, -1):
                if not np.isnan(x_coords[i]) and not np.isnan(y_coords[i]):
                    before_idx = i
                    break
            
            # Look for valid point after
            for i in range(end_idx + 1, len(data_points)):
                if not np.isnan(x_coords[i]) and not np.isnan(y_coords[i]):
                    after_idx = i
                    break
            
            # Interpolate if we have valid points on both sides
            if before_idx is not None and after_idx is not None:
                # Linear interpolation
                for i, idx in enumerate(segment):
                    # Calculate interpolation factor
                    factor = (times[idx] - times[before_idx]) / (times[after_idx] - times[before_idx])
                    
                    # Interpolate coordinates
                    x_coords[idx] = x_coords[before_idx] + factor * (x_coords[after_idx] - x_coords[before_idx])
                    y_coords[idx] = y_coords[before_idx] + factor * (y_coords[after_idx] - y_coords[before_idx])
            
            elif before_idx is not None:
                # Only valid point before - use last known position
                self.logger.warning(f"No valid point after segment, using last known position for indices {segment}")
                for idx in segment:
                    x_coords[idx] = x_coords[before_idx]
                    y_coords[idx] = y_coords[before_idx]
            
            elif after_idx is not None:
                # Only valid point after - use next known position
                self.logger.warning(f"No valid point before segment, using next known position for indices {segment}")
                for idx in segment:
                    x_coords[idx] = x_coords[after_idx]
                    y_coords[idx] = y_coords[after_idx]
            
            else:
                # No valid points found - cannot interpolate
                self.logger.error(f"Cannot interpolate segment {segment} - no valid reference points found")
        
        # Update the data points with interpolated values
        for i, point in enumerate(data_points):
            data_points[i] = TrackingPoint(
                time=point.time,
                arena=point.arena,
                track=point.track,
                x=float(x_coords[i]),
                y=float(y_coords[i]),
                label=point.label,
                video_seq=point.video_seq
            )
        
        self.logger.info(f"Successfully interpolated {interpolated_count} missing data points")
        return data_points
    
    @log_function_call(logger)
    def load_multiple_sessions(self, directories: List[Path]) -> List[TrackingSession]:
        """
        Load tracking data from multiple directories.
        
        Args:
            directories: List of directories to search for tracking files
            
        Returns:
            List of TrackingSession objects
        """
        all_sessions = []
        
        for directory in directories:
            self.logger.info(f"Processing directory: {directory}")
            
            try:
                tracking_files = self.find_tracking_files(directory)
                
                for file_path in tracking_files:
                    try:
                        session = self.load_tracking_data(file_path)
                        all_sessions.append(session)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to load {file_path}: {e}"
                        )
                        continue
                        
            except Exception as e:
                self.logger.error(f"Error processing directory {directory}: {e}")
                continue
        
        self.logger.info(f"Successfully loaded {len(all_sessions)} sessions")
        return all_sessions
    
    @log_function_call(logger)
    def create_standard_epm_zones(self, tracking_session: TrackingSession) -> List[BoundingBox]:
        """
        Create standard EPM (Elevated Plus Maze) zones based on tracking data bounds.
        
        This creates a basic zone layout for EPM analysis:
        - Center zone (intersection)
        - Open arms (typically horizontal)
        - Closed arms (typically vertical)
        
        Args:
            tracking_session: Tracking session to analyze
            
        Returns:
            List of BoundingBox objects representing EPM zones
        """
        df = tracking_session.get_dataframe()
        
        # Calculate bounds of the tracking area
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # Define zone dimensions (these are rough estimates)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        center_width = x_range * 0.2  # Center is 20% of total width
        center_height = y_range * 0.2  # Center is 20% of total height
        
        zones = [
            BoundingBox(
                name="center",
                x_min=x_center - center_width/2,
                x_max=x_center + center_width/2,
                y_min=y_center - center_height/2,
                y_max=y_center + center_height/2,
                description="Center intersection of EPM"
            ),
            BoundingBox(
                name="open_arm_east",
                x_min=x_center + center_width/2,
                x_max=x_max,
                y_min=y_center - center_height/2,
                y_max=y_center + center_height/2,
                description="Eastern open arm"
            ),
            BoundingBox(
                name="open_arm_west", 
                x_min=x_min,
                x_max=x_center - center_width/2,
                y_min=y_center - center_height/2,
                y_max=y_center + center_height/2,
                description="Western open arm"
            ),
            BoundingBox(
                name="closed_arm_north",
                x_min=x_center - center_width/2,
                x_max=x_center + center_width/2,
                y_min=y_min,
                y_max=y_center - center_height/2,
                description="Northern closed arm"
            ),
            BoundingBox(
                name="closed_arm_south",
                x_min=x_center - center_width/2,
                x_max=x_center + center_width/2,
                y_min=y_center + center_height/2,
                y_max=y_max,
                description="Southern closed arm"
            )
        ]
        
        self.logger.info(f"Created {len(zones)} standard EPM zones")
        return zones

    @log_function_call(logger)
    def create_standard_epm_zones_from_multiple_sessions(self, sessions: List[TrackingSession]) -> List[BoundingBox]:
        """
        Create standard EPM zones based on tracking data bounds from multiple sessions.
        
        Args:
            sessions: List of tracking sessions to analyze
            
        Returns:
            List of BoundingBox objects representing EPM zones
        """
        if not sessions:
            raise ValueError("At least one tracking session is required")
        
        self.logger.info(f"Creating EPM zones from {len(sessions)} sessions")
        
        # Find global bounds across ALL sessions
        global_x_min = float('inf')
        global_x_max = float('-inf')
        global_y_min = float('inf')
        global_y_max = float('-inf')
        
        for session in sessions:
            df = session.get_dataframe()
            session_x_min, session_x_max = df['x'].min(), df['x'].max()
            session_y_min, session_y_max = df['y'].min(), df['y'].max()
            
            global_x_min = min(global_x_min, session_x_min)
            global_x_max = max(global_x_max, session_x_max)
            global_y_min = min(global_y_min, session_y_min)
            global_y_max = max(global_y_max, session_y_max)
        
        x_center = (global_x_min + global_x_max) / 2
        y_center = (global_y_min + global_y_max) / 2
        
        # Define zone dimensions (these are rough estimates)
        x_range = global_x_max - global_x_min
        y_range = global_y_max - global_y_min
        
        center_width = x_range * 0.2  # Center is 20% of total width
        center_height = y_range * 0.2  # Center is 20% of total height
        
        zones = [
            BoundingBox(
                name="center",
                x_min=x_center - center_width/2,
                x_max=x_center + center_width/2,
                y_min=y_center - center_height/2,
                y_max=y_center + center_height/2,
                description="Center intersection of EPM"
            ),
            BoundingBox(
                name="open_arm_east",
                x_min=x_center + center_width/2,
                x_max=global_x_max,
                y_min=y_center - center_height/2,
                y_max=y_center + center_height/2,
                description="Eastern open arm"
            ),
            BoundingBox(
                name="open_arm_west", 
                x_min=global_x_min,
                x_max=x_center - center_width/2,
                y_min=y_center - center_height/2,
                y_max=y_center + center_height/2,
                description="Western open arm"
            ),
            BoundingBox(
                name="closed_arm_north",
                x_min=x_center - center_width/2,
                x_max=x_center + center_width/2,
                y_min=global_y_min,
                y_max=y_center - center_height/2,
                description="Northern closed arm"
            ),
            BoundingBox(
                name="closed_arm_south",
                x_min=x_center - center_width/2,
                x_max=x_center + center_width/2,
                y_min=y_center + center_height/2,
                y_max=global_y_max,
                description="Southern closed arm"
            )
        ]
        
        self.logger.info(f"Created {len(zones)} standard EPM zones from multiple sessions")
        return zones
    
    @log_function_call(logger)
    def create_open_field_zones(self, tracking_session: TrackingSession) -> List[BoundingBox]:
        """
        Create standard Open Field zones based on tracking data bounds.
        
        Creates concentric zones:
        - Center zone (inner area)
        - Middle zone (middle ring)
        - Peripheral zone (outer ring)
        
        Args:
            tracking_session: Tracking session to analyze
            
        Returns:
            List of BoundingBox objects representing Open Field zones
        """
        df = tracking_session.get_dataframe()
        
        # Calculate bounds of the tracking area
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # Define zone dimensions
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Center zone (inner 33%)
        center_width = x_range * 0.33
        center_height = y_range * 0.33
        
        # Middle zone (middle 33%)
        middle_width = x_range * 0.66
        middle_height = y_range * 0.66
        
        zones = [
            BoundingBox(
                name="center",
                x_min=x_center - center_width/2,
                x_max=x_center + center_width/2,
                y_min=y_center - center_height/2,
                y_max=y_center + center_height/2,
                description="Center zone (inner 33%)"
            ),
            BoundingBox(
                name="middle",
                x_min=x_center - middle_width/2,
                x_max=x_center + middle_width/2,
                y_min=y_center - middle_height/2,
                y_max=y_center + middle_height/2,
                description="Middle zone (middle 33%)"
            ),
            BoundingBox(
                name="peripheral",
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                description="Peripheral zone (entire area)"
            )
        ]
        
        self.logger.info(f"Created {len(zones)} standard Open Field zones")
        return zones

    @log_function_call(logger)
    def create_open_field_zones_from_multiple_sessions(self, sessions: List[TrackingSession]) -> List[BoundingBox]:
        """
        Create standard Open Field zones based on tracking data bounds from multiple sessions.
        
        Args:
            sessions: List of tracking sessions to analyze
            
        Returns:
            List of BoundingBox objects representing Open Field zones
        """
        if not sessions:
            raise ValueError("At least one tracking session is required")
        
        self.logger.info(f"Creating Open Field zones from {len(sessions)} sessions")
        
        # Find global bounds across ALL sessions
        global_x_min = float('inf')
        global_x_max = float('-inf')
        global_y_min = float('inf')
        global_y_max = float('-inf')
        
        for session in sessions:
            df = session.get_dataframe()
            session_x_min, session_x_max = df['x'].min(), df['x'].max()
            session_y_min, session_y_max = df['y'].min(), df['y'].max()
            
            global_x_min = min(global_x_min, session_x_min)
            global_x_max = max(global_x_max, session_x_max)
            global_y_min = min(global_y_min, session_y_min)
            global_y_max = max(global_y_max, session_y_max)
        
        x_center = (global_x_min + global_x_max) / 2
        y_center = (global_y_min + global_y_max) / 2
        
        # Define zone dimensions
        x_range = global_x_max - global_x_min
        y_range = global_y_max - global_y_min
        
        # Center zone (inner 33%)
        center_width = x_range * 0.33
        center_height = y_range * 0.33
        
        # Middle zone (middle 33%)
        middle_width = x_range * 0.66
        middle_height = y_range * 0.66
        
        zones = [
            BoundingBox(
                name="center",
                x_min=x_center - center_width/2,
                x_max=x_center + center_width/2,
                y_min=y_center - center_height/2,
                y_max=y_center + center_height/2,
                description="Center zone (inner 33%)"
            ),
            BoundingBox(
                name="middle",
                x_min=x_center - middle_width/2,
                x_max=x_center + middle_width/2,
                y_min=y_center - middle_height/2,
                y_max=y_center + middle_height/2,
                description="Middle zone (33-66%)"
            ),
            BoundingBox(
                name="peripheral",
                x_min=global_x_min,
                x_max=global_x_max,
                y_min=global_y_min,
                y_max=global_y_max,
                description="Peripheral zone (complete area)"
            )
        ]
        
        self.logger.info(f"Created {len(zones)} standard Open Field zones from multiple sessions")
        return zones
    
    @log_function_call(logger)
    def create_oft_zones(self, tracking_session: TrackingSession, inner_zone_ratio: float = 0.6) -> List[BoundingBox]:
        """
        Create OFT (Open Field Test) zones based on tracking data bounds.
        
        Creates zones for a mouse in a container:
        - Inner zone: Centered inner area (size determined by inner_zone_ratio)
        - Outer zone: Complete container area (ensures NO points are outside)
        
        Args:
            tracking_session: Tracking session to analyze
            inner_zone_ratio: Ratio of inner zone size to outer zone size (0.0-1.0)
            
        Returns:
            List of BoundingBox objects representing OFT zones
            
        Raises:
            ValueError: If inner_zone_ratio is not between 0.0 and 1.0
        """
        if not 0.0 <= inner_zone_ratio <= 1.0:
            raise ValueError(f"inner_zone_ratio must be between 0.0 and 1.0, got {inner_zone_ratio}")
        
        df = tracking_session.get_dataframe()
        
        # Calculate bounds of the tracking area
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        
        # Add padding to ensure NO points are outside the outer zone
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        x_padding = x_range * 0.1  # 10% padding for safety
        y_padding = y_range * 0.1  # 10% padding for safety
        
        # **NEW APPROACH: Use actual data bounds directly**
        # Outer zone covers ALL data with padding
        outer_x_min = x_min - x_padding
        outer_x_max = x_max + x_padding
        outer_y_min = y_min - y_padding
        outer_y_max = y_max + y_padding
        
        # Calculate center for inner zone
        x_center = (outer_x_min + outer_x_max) / 2
        y_center = (outer_y_min + outer_y_max) / 2
        
        # Inner zone is centered within the outer zone
        outer_width = outer_x_max - outer_x_min
        outer_height = outer_y_max - outer_y_min
        
        inner_width = outer_width * inner_zone_ratio
        inner_height = outer_height * inner_zone_ratio
        
        inner_x_min = x_center - inner_width / 2
        inner_x_max = x_center + inner_width / 2
        inner_y_min = y_center - inner_height / 2
        inner_y_max = y_center + inner_height / 2
        
        zones = [
            BoundingBox(
                name="inner",
                x_min=inner_x_min,
                x_max=inner_x_max,
                y_min=inner_y_min,
                y_max=inner_y_max,
                description=f"Inner zone ({inner_zone_ratio*100:.1f}% of outer zone area)"
            ),
            BoundingBox(
                name="outer",
                x_min=outer_x_min,
                x_max=outer_x_max,
                y_min=outer_y_min,
                y_max=outer_y_max,
                description="Outer zone (covers all tracking data with padding)"
            )
        ]
        
        # Verify that all points are inside the outer zone
        points_outside = df[
            (df['x'] < outer_x_min) | (df['x'] > outer_x_max) |
            (df['y'] < outer_y_min) | (df['y'] > outer_y_max)
        ]
        
        if len(points_outside) > 0:
            self.logger.error(f"CRITICAL: {len(points_outside)} points outside outer zone - expanding with emergency padding")
            
            # Emergency expansion - use absolute bounds with more padding
            emergency_padding = max(x_range, y_range) * 0.2  # 20% emergency padding
            outer_x_min = x_min - emergency_padding
            outer_x_max = x_max + emergency_padding
            outer_y_min = y_min - emergency_padding
            outer_y_max = y_max + emergency_padding
            
            # Recalculate inner zone
            x_center = (outer_x_min + outer_x_max) / 2
            y_center = (outer_y_min + outer_y_max) / 2
            outer_width = outer_x_max - outer_x_min
            outer_height = outer_y_max - outer_y_min
            inner_width = outer_width * inner_zone_ratio
            inner_height = outer_height * inner_zone_ratio
            
            inner_x_min = x_center - inner_width / 2
            inner_x_max = x_center + inner_width / 2
            inner_y_min = y_center - inner_height / 2
            inner_y_max = y_center + inner_height / 2
            
            # Update zones
            zones = [
                BoundingBox(
                    name="inner",
                    x_min=inner_x_min,
                    x_max=inner_x_max,
                    y_min=inner_y_min,
                    y_max=inner_y_max,
                    description=f"Inner zone ({inner_zone_ratio*100:.1f}% of outer zone area) - emergency expanded"
                ),
                BoundingBox(
                    name="outer",
                    x_min=outer_x_min,
                    x_max=outer_x_max,
                    y_min=outer_y_min,
                    y_max=outer_y_max,
                    description="Outer zone (emergency expanded to cover all data)"
                )
            ]
        
        self.logger.info(f"Created {len(zones)} OFT zones")
        self.logger.info(f"Outer zone: ({outer_x_min:.1f}, {outer_y_min:.1f}) to ({outer_x_max:.1f}, {outer_y_max:.1f}) [{outer_width:.1f} × {outer_height:.1f} mm]")
        self.logger.info(f"Inner zone: ({inner_x_min:.1f}, {inner_y_min:.1f}) to ({inner_x_max:.1f}, {inner_y_max:.1f}) [{inner_width:.1f} × {inner_height:.1f} mm]")
        
        return zones

    @log_function_call(logger)
    def create_oft_zones_from_multiple_sessions(self, sessions: List[TrackingSession], inner_zone_ratio: float = 0.6) -> List[BoundingBox]:
        """
        Create OFT zones based on data bounds from multiple tracking sessions.
        
        This ensures that zones encompass ALL data points from ALL sessions,
        preventing the issue where some sessions have 0% time in zones.
        
        Args:
            sessions: List of tracking sessions to analyze
            inner_zone_ratio: Ratio of inner zone size to outer zone size (0.0-1.0)
            
        Returns:
            List of BoundingBox objects representing OFT zones
            
        Raises:
            ValueError: If inner_zone_ratio is not between 0.0 and 1.0 or no sessions provided
        """
        if not sessions:
            raise ValueError("At least one tracking session is required")
        
        if not 0.0 <= inner_zone_ratio <= 1.0:
            raise ValueError(f"inner_zone_ratio must be between 0.0 and 1.0, got {inner_zone_ratio}")
        
        self.logger.info(f"Creating OFT zones from {len(sessions)} sessions")
        
        # Find global bounds across ALL sessions
        global_x_min = float('inf')
        global_x_max = float('-inf')
        global_y_min = float('inf')
        global_y_max = float('-inf')
        total_points = 0
        
        for session in sessions:
            df = session.get_dataframe()
            session_x_min, session_x_max = df['x'].min(), df['x'].max()
            session_y_min, session_y_max = df['y'].min(), df['y'].max()
            
            global_x_min = min(global_x_min, session_x_min)
            global_x_max = max(global_x_max, session_x_max)
            global_y_min = min(global_y_min, session_y_min)
            global_y_max = max(global_y_max, session_y_max)
            total_points += len(df)
            
            self.logger.info(f"Session {session.session_name}: X=({session_x_min:.1f}, {session_x_max:.1f}), Y=({session_y_min:.1f}, {session_y_max:.1f})")
        
        self.logger.info(f"Global bounds across all sessions: X=({global_x_min:.1f}, {global_x_max:.1f}), Y=({global_y_min:.1f}, {global_y_max:.1f})")
        self.logger.info(f"Total data points across all sessions: {total_points}")
        
        # Add padding to ensure NO points are outside the outer zone
        x_range = global_x_max - global_x_min
        y_range = global_y_max - global_y_min
        
        x_padding = x_range * 0.1  # 10% padding for safety
        y_padding = y_range * 0.1  # 10% padding for safety
        
        # **NEW APPROACH: Use actual data bounds directly instead of forced squares**
        # Outer zone covers ALL data with padding
        outer_x_min = global_x_min - x_padding
        outer_x_max = global_x_max + x_padding
        outer_y_min = global_y_min - y_padding
        outer_y_max = global_y_max + y_padding
        
        # Calculate center for inner zone
        x_center = (outer_x_min + outer_x_max) / 2
        y_center = (outer_y_min + outer_y_max) / 2
        
        # Inner zone is a centered rectangle within the outer zone
        outer_width = outer_x_max - outer_x_min
        outer_height = outer_y_max - outer_y_min
        
        inner_width = outer_width * inner_zone_ratio
        inner_height = outer_height * inner_zone_ratio
        
        inner_x_min = x_center - inner_width / 2
        inner_x_max = x_center + inner_width / 2
        inner_y_min = y_center - inner_height / 2
        inner_y_max = y_center + inner_height / 2
        
        zones = [
            BoundingBox(
                name="inner",
                x_min=inner_x_min,
                x_max=inner_x_max,
                y_min=inner_y_min,
                y_max=inner_y_max,
                description=f"Inner zone ({inner_zone_ratio*100:.1f}% of outer zone area)"
            ),
            BoundingBox(
                name="outer",
                x_min=outer_x_min,
                x_max=outer_x_max,
                y_min=outer_y_min,
                y_max=outer_y_max,
                description="Outer zone (covers all tracking data with padding)"
            )
        ]
        
        # Verify that all points from all sessions are inside the outer zone
        points_outside_total = 0
        points_inside_outer = 0
        points_inside_inner = 0
        
        for session in sessions:
            df = session.get_dataframe()
            
            # Check outer zone coverage
            points_outside = df[
                (df['x'] < outer_x_min) | (df['x'] > outer_x_max) |
                (df['y'] < outer_y_min) | (df['y'] > outer_y_max)
            ]
            points_outside_total += len(points_outside)
            
            # Count points inside zones for verification
            session_points_inside_outer = len(df[
                (df['x'] >= outer_x_min) & (df['x'] <= outer_x_max) &
                (df['y'] >= outer_y_min) & (df['y'] <= outer_y_max)
            ])
            
            session_points_inside_inner = len(df[
                (df['x'] >= inner_x_min) & (df['x'] <= inner_x_max) &
                (df['y'] >= inner_y_min) & (df['y'] <= inner_y_max)
            ])
            
            points_inside_outer += session_points_inside_outer
            points_inside_inner += session_points_inside_inner
            
            self.logger.info(f"Session {session.session_name}: {session_points_inside_outer}/{len(df)} points in outer zone ({session_points_inside_outer/len(df)*100:.1f}%)")
            self.logger.info(f"Session {session.session_name}: {session_points_inside_inner}/{len(df)} points in inner zone ({session_points_inside_inner/len(df)*100:.1f}%)")
            
            if len(points_outside) > 0:
                self.logger.error(f"❌ Session {session.session_name}: {len(points_outside)} points outside outer zone!")
            
        if points_outside_total > 0:
            self.logger.error(f"CRITICAL: {points_outside_total} total points are outside the outer zone!")
            self.logger.error("This indicates a bug in the algorithm - expanding with emergency padding...")
            
            # Emergency expansion
            emergency_padding = max(x_range, y_range) * 0.2  # 20% emergency padding
            outer_x_min = global_x_min - emergency_padding
            outer_x_max = global_x_max + emergency_padding
            outer_y_min = global_y_min - emergency_padding
            outer_y_max = global_y_max + emergency_padding
            
            # Recalculate inner zone
            x_center = (outer_x_min + outer_x_max) / 2
            y_center = (outer_y_min + outer_y_max) / 2
            outer_width = outer_x_max - outer_x_min
            outer_height = outer_y_max - outer_y_min
            inner_width = outer_width * inner_zone_ratio
            inner_height = outer_height * inner_zone_ratio
            
            inner_x_min = x_center - inner_width / 2
            inner_x_max = x_center + inner_width / 2
            inner_y_min = y_center - inner_height / 2
            inner_y_max = y_center + inner_height / 2
            
            # Update zones
            zones = [
                BoundingBox(
                    name="inner",
                    x_min=inner_x_min,
                    x_max=inner_x_max,
                    y_min=inner_y_min,
                    y_max=inner_y_max,
                    description=f"Inner zone ({inner_zone_ratio*100:.1f}% of outer zone area) - emergency expanded"
                ),
                BoundingBox(
                    name="outer",
                    x_min=outer_x_min,
                    x_max=outer_x_max,
                    y_min=outer_y_min,
                    y_max=outer_y_max,
                    description="Outer zone (emergency expanded to cover all data)"
                )
            ]
        
        self.logger.info(f"Created {len(zones)} OFT zones with inner ratio {inner_zone_ratio}")
        self.logger.info(f"Outer zone: ({outer_x_min:.1f}, {outer_y_min:.1f}) to ({outer_x_max:.1f}, {outer_y_max:.1f}) [{outer_width:.1f} × {outer_height:.1f} mm]")
        self.logger.info(f"Inner zone: ({inner_x_min:.1f}, {inner_y_min:.1f}) to ({inner_x_max:.1f}, {inner_y_max:.1f}) [{inner_width:.1f} × {inner_height:.1f} mm]")
        self.logger.info(f"Zone coverage verification: {points_inside_outer}/{total_points} points in outer zone, {points_inside_inner}/{total_points} points in inner zone")
        
        if points_outside_total == 0:
            self.logger.info("✅ All tracking points are within the outer zone")
        else:
            self.logger.error(f"❌ {points_outside_total} points still outside outer zone - algorithm needs further debugging!")
        
        return zones

    @log_function_call(logger)
    def create_individual_oft_zones_for_each_session(
        self, 
        sessions: List[TrackingSession], 
        inner_zone_ratio: float = 0.6
    ) -> List[BoundingBox]:
        """
        Create individual OFT zones for each tracking session separately.
        
        This method creates separate inner and outer zones for each session:
        - If you have 2 sessions, you get 4 zones: inner_1, outer_1, inner_2, outer_2
        - If you have 3 sessions, you get 6 zones: inner_1, outer_1, inner_2, outer_2, inner_3, outer_3
        
        Args:
            sessions: List of tracking sessions to analyze
            inner_zone_ratio: Ratio of inner zone size to outer zone size (0.0-1.0)
            
        Returns:
            List of BoundingBox objects representing individual OFT zones for each session
            
        Raises:
            ValueError: If inner_zone_ratio is not between 0.0 and 1.0
        """
        if not 0.0 <= inner_zone_ratio <= 1.0:
            raise ValueError(f"inner_zone_ratio must be between 0.0 and 1.0, got {inner_zone_ratio}")
        
        if not sessions:
            raise ValueError("Need at least one session to create individual OFT zones")
        
        self.logger.info(f"Creating individual OFT zones for {len(sessions)} sessions with inner ratio {inner_zone_ratio}")
        
        all_zones = []
        
        for i, session in enumerate(sessions, 1):
            self.logger.info(f"Creating zones for session {i}: {session.session_name}")
            
            # Create zones for this individual session
            session_zones = self.create_oft_zones(session, inner_zone_ratio)
            
            # Rename zones to include session identifier
            for zone in session_zones:
                # Create new zone with session-specific name
                session_zone = BoundingBox(
                    name=f"{zone.name}_{i}",
                    x_min=zone.x_min,
                    x_max=zone.x_max,
                    y_min=zone.y_min,
                    y_max=zone.y_max,
                    description=f"{zone.description} - Session {i} ({session.session_name})"
                )
                all_zones.append(session_zone)
        
        self.logger.info(f"Created {len(all_zones)} individual OFT zones: {[z.name for z in all_zones]}")
        
        # Log zone summary
        for i, session in enumerate(sessions, 1):
            session_zones = [z for z in all_zones if z.name.endswith(f"_{i}")]
            self.logger.info(f"Session {i} ({session.session_name}) zones:")
            for zone in session_zones:
                width = zone.x_max - zone.x_min
                height = zone.y_max - zone.y_min
                self.logger.info(f"  {zone.name}: ({zone.x_min:.1f}, {zone.y_min:.1f}) to ({zone.x_max:.1f}, {zone.y_max:.1f}) [{width:.1f} × {height:.1f} mm]")
        
        return all_zones

    @log_function_call(logger)
    def create_individual_oft_zones_for_each_session(self, sessions: List[TrackingSession], inner_zone_ratio: float = 0.6) -> List[BoundingBox]:
        """
        Create separate OFT zones for each individual tracking session.
        
        This creates separate inner/outer zone pairs for each session:
        - For Tracking_RealSpace_1: creates 'inner_1' and 'outer_1' zones
        - For Tracking_RealSpace_2: creates 'inner_2' and 'outer_2' zones
        - And so on...
        
        Each zone pair is optimized for its specific session's data bounds.
        
        Args:
            sessions: List of tracking sessions to analyze
            inner_zone_ratio: Ratio of inner zone size to outer zone size (0.0-1.0)
            
        Returns:
            List of BoundingBox objects representing individual OFT zones for each session
            
        Raises:
            ValueError: If inner_zone_ratio is not between 0.0 and 1.0 or no sessions provided
        """
        if not sessions:
            raise ValueError("At least one tracking session is required")
        
        if not 0.0 <= inner_zone_ratio <= 1.0:
            raise ValueError(f"inner_zone_ratio must be between 0.0 and 1.0, got {inner_zone_ratio}")
        
        self.logger.info(f"Creating individual OFT zones for {len(sessions)} sessions (ratio: {inner_zone_ratio})")
        
        all_zones = []
        
        for i, session in enumerate(sessions, 1):
            session_suffix = f"_{i}"
            
            # Extract session number from filename if possible
            session_name = session.session_name
            if "RealSpace" in session_name:
                # Try to extract number from "Tracking_RealSpace_X" format
                try:
                    parts = session_name.split("_")
                    for part in parts:
                        if part.isdigit():
                            session_suffix = f"_{part}"
                            break
                except:
                    pass
            
            self.logger.info(f"Creating zones for session: {session_name} (suffix: {session_suffix})")
            
            # Get data for this specific session
            df = session.get_dataframe()
            
            if len(df) == 0:
                self.logger.warning(f"Session {session_name} has no data points, skipping")
                continue
            
            # Find bounds for this session only
            x_min = df['x'].min()
            x_max = df['x'].max()
            y_min = df['y'].min()
            y_max = df['y'].max()
            
            self.logger.info(f"Session {session_name} bounds: X=({x_min:.1f}, {x_max:.1f}), Y=({y_min:.1f}, {y_max:.1f})")
            
            # Add padding to ensure NO points are outside the outer zone for this session
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            x_padding = x_range * 0.1  # 10% padding for safety
            y_padding = y_range * 0.1  # 10% padding for safety
            
            # Outer zone covers this session's data with padding
            outer_x_min = x_min - x_padding
            outer_x_max = x_max + x_padding
            outer_y_min = y_min - y_padding
            outer_y_max = y_max + y_padding
            
            # Calculate center for inner zone
            x_center = (outer_x_min + outer_x_max) / 2
            y_center = (outer_y_min + outer_y_max) / 2
            
            # Inner zone is a centered rectangle within the outer zone
            outer_width = outer_x_max - outer_x_min
            outer_height = outer_y_max - outer_y_min
            
            inner_width = outer_width * inner_zone_ratio
            inner_height = outer_height * inner_zone_ratio
            
            inner_x_min = x_center - inner_width / 2
            inner_x_max = x_center + inner_width / 2
            inner_y_min = y_center - inner_height / 2
            inner_y_max = y_center + inner_height / 2
            
            # Create zones for this session
            session_zones = [
                BoundingBox(
                    name=f"inner{session_suffix}",
                    x_min=inner_x_min,
                    x_max=inner_x_max,
                    y_min=inner_y_min,
                    y_max=inner_y_max,
                    description=f"Inner zone for {session_name} ({inner_zone_ratio*100:.1f}% of outer zone area)"
                ),
                BoundingBox(
                    name=f"outer{session_suffix}",
                    x_min=outer_x_min,
                    x_max=outer_x_max,
                    y_min=outer_y_min,
                    y_max=outer_y_max,
                    description=f"Outer zone for {session_name} (covers all tracking data with padding)"
                )
            ]
            
            # Verify that all points from this session are inside its outer zone
            points_outside = df[
                (df['x'] < outer_x_min) | (df['x'] > outer_x_max) |
                (df['y'] < outer_y_min) | (df['y'] > outer_y_max)
            ]
            
            if len(points_outside) > 0:
                self.logger.error(f"CRITICAL: {len(points_outside)} points from {session_name} are outside its outer zone!")
                
                # Emergency expansion
                emergency_padding = max(x_range, y_range) * 0.2  # 20% emergency padding
                outer_x_min = x_min - emergency_padding
                outer_x_max = x_max + emergency_padding
                outer_y_min = y_min - emergency_padding
                outer_y_max = y_max + emergency_padding
                
                # Recalculate inner zone
                x_center = (outer_x_min + outer_x_max) / 2
                y_center = (outer_y_min + outer_y_max) / 2
                outer_width = outer_x_max - outer_x_min
                outer_height = outer_y_max - outer_y_min
                inner_width = outer_width * inner_zone_ratio
                inner_height = outer_height * inner_zone_ratio
                
                inner_x_min = x_center - inner_width / 2
                inner_x_max = x_center + inner_width / 2
                inner_y_min = y_center - inner_height / 2
                inner_y_max = y_center + inner_height / 2
                
                # Update zones with emergency expansion
                session_zones = [
                    BoundingBox(
                        name=f"inner{session_suffix}",
                        x_min=inner_x_min,
                        x_max=inner_x_max,
                        y_min=inner_y_min,
                        y_max=inner_y_max,
                        description=f"Inner zone for {session_name} ({inner_zone_ratio*100:.1f}% of outer zone area) - emergency expanded"
                    ),
                    BoundingBox(
                        name=f"outer{session_suffix}",
                        x_min=outer_x_min,
                        x_max=outer_x_max,
                        y_min=outer_y_min,
                        y_max=outer_y_max,
                        description=f"Outer zone for {session_name} (emergency expanded to cover all data)"
                    )
                ]
                
                self.logger.warning(f"Applied emergency expansion for {session_name}")
            
            # Verify coverage after creation
            points_inside_outer = len(df[
                (df['x'] >= outer_x_min) & (df['x'] <= outer_x_max) &
                (df['y'] >= outer_y_min) & (df['y'] <= outer_y_max)
            ])
            
            points_inside_inner = len(df[
                (df['x'] >= inner_x_min) & (df['x'] <= inner_x_max) &
                (df['y'] >= inner_y_min) & (df['y'] <= inner_y_max)
            ])
            
            self.logger.info(f"Session {session_name}: {points_inside_outer}/{len(df)} points in outer zone ({points_inside_outer/len(df)*100:.1f}%)")
            self.logger.info(f"Session {session_name}: {points_inside_inner}/{len(df)} points in inner zone ({points_inside_inner/len(df)*100:.1f}%)")
            
            # Add zones to the total list
            all_zones.extend(session_zones)
            
            self.logger.info(f"Created zones for {session_name}: {[z.name for z in session_zones]}")
            self.logger.info(f"  Outer zone: ({outer_x_min:.1f}, {outer_y_min:.1f}) to ({outer_x_max:.1f}, {outer_y_max:.1f}) [{outer_width:.1f} × {outer_height:.1f} mm]")
            self.logger.info(f"  Inner zone: ({inner_x_min:.1f}, {inner_y_min:.1f}) to ({inner_x_max:.1f}, {inner_y_max:.1f}) [{inner_width:.1f} × {inner_height:.1f} mm]")
        
        self.logger.info(f"Created {len(all_zones)} total zones for {len(sessions)} sessions")
        self.logger.info(f"Zone names: {[z.name for z in all_zones]}")
        
        return all_zones
