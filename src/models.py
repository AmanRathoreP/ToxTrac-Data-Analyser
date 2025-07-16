"""
Core data models and classes for ToxTrac data analysis.

This module contains the core data structures used throughout the application
for representing tracking data, zones, and analysis results.

Author: Aman Rathore
Contact: amanr.me | amanrathore9753 <at> gmail <dot> com
Created on: Monday, July 14, 2025 at 10:40
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import pandas as pd


@dataclass
class TrackingPoint:
    """
    Represents a single tracking point in time.
    
    Attributes:
        time: Time in seconds
        arena: Arena identifier
        track: Track identifier  
        x: X coordinate in mm
        y: Y coordinate in mm
        label: Label identifier
        video_seq: Video sequence number (optional)
    """
    time: float
    arena: int
    track: int
    x: float
    y: float
    label: int
    video_seq: Optional[int] = None


@dataclass
class BoundingBox:
    """
    Represents a rectangular bounding box/zone.
    
    Attributes:
        name: Human-readable name for the zone
        x_min: Minimum X coordinate
        y_min: Minimum Y coordinate  
        x_max: Maximum X coordinate
        y_max: Maximum Y coordinate
        description: Optional description
    """
    name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    description: Optional[str] = None
    
    def contains_point(self, x: float, y: float) -> bool:
        """
        Check if a point is inside this bounding box.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if point is inside the bounding box, False otherwise
        """
        return (self.x_min <= x <= self.x_max and 
                self.y_min <= y <= self.y_max)


@dataclass
class TrackingSession:
    """
    Represents a complete tracking session with all data points.
    
    Attributes:
        session_name: Name/identifier for this session
        data_points: List of all tracking points
        source_file: Path to the source data file
        total_duration: Total duration in seconds
        sampling_rate: Approximate sampling rate in Hz
    """
    session_name: str
    data_points: List[TrackingPoint]
    source_file: Path
    total_duration: float
    sampling_rate: float
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Convert tracking data to pandas DataFrame for easier analysis.
        
        Returns:
            DataFrame with all tracking points
        """
        data = []
        for point in self.data_points:
            row = {
                'time': point.time,
                'arena': point.arena,
                'track': point.track,
                'x': point.x,
                'y': point.y,
                'label': point.label
            }
            if point.video_seq is not None:
                row['video_seq'] = point.video_seq
            data.append(row)
        
        return pd.DataFrame(data)


@dataclass 
class ZoneAnalysisResult:
    """
    Results from analyzing time spent in a specific zone.
    
    Attributes:
        zone_name: Name of the analyzed zone
        total_time_inside: Total time spent inside the zone (seconds)
        total_time_outside: Total time spent outside the zone (seconds)
        percentage_inside: Percentage of time spent inside (0-100)
        entry_count: Number of times the animal entered the zone
        exit_count: Number of times the animal exited the zone
        mean_visit_duration: Average duration of visits to the zone
        longest_visit: Longest continuous visit duration
        shortest_visit: Shortest continuous visit duration
        distance_inside: Total distance traveled while inside the zone (mm)
        distance_outside: Total distance traveled while outside the zone (mm)
        mean_speed_inside: Average speed while inside the zone (mm/s)
        mean_speed_outside: Average speed while outside the zone (mm/s)
    """
    zone_name: str
    total_time_inside: float
    total_time_outside: float
    percentage_inside: float
    entry_count: int
    exit_count: int
    mean_visit_duration: float
    longest_visit: float
    shortest_visit: float
    distance_inside: float = 0.0
    distance_outside: float = 0.0
    mean_speed_inside: float = 0.0
    mean_speed_outside: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for easy serialization."""
        return {
            'zone_name': self.zone_name,
            'total_time_inside_sec': self.total_time_inside,
            'total_time_outside_sec': self.total_time_outside,
            'percentage_inside': self.percentage_inside,
            'entry_count': self.entry_count,
            'exit_count': self.exit_count,
            'mean_visit_duration_sec': self.mean_visit_duration,
            'longest_visit_sec': self.longest_visit,
            'shortest_visit_sec': self.shortest_visit,
            'distance_inside_mm': self.distance_inside,
            'distance_outside_mm': self.distance_outside,
            'mean_speed_inside_mm_per_sec': self.mean_speed_inside,
            'mean_speed_outside_mm_per_sec': self.mean_speed_outside
        }


@dataclass
class SessionAnalysisResult:
    """
    Complete analysis results for a tracking session.
    
    Attributes:
        session_name: Name of the analyzed session
        zone_results: List of zone analysis results
        total_duration: Total session duration
        total_distance: Total distance traveled (if calculated)
        mean_speed: Mean speed (if calculated)
    """
    session_name: str
    zone_results: List[ZoneAnalysisResult]
    total_duration: float
    total_distance: Optional[float] = None
    mean_speed: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for easy serialization."""
        return {
            'session_name': self.session_name,
            'total_duration_sec': self.total_duration,
            'total_distance_mm': self.total_distance,
            'mean_speed_mm_per_sec': self.mean_speed,
            'zone_results': [zone.to_dict() for zone in self.zone_results]
        }
