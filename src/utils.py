"""
Utility functions for ToxTrac Data Analyzer.

This module contains helper functions and utilities used throughout
the application.

Author: Aman Rathore
Contact: amanr.me | amanrathore9753 <at> gmail <dot> com
Created on: Monday, July 14, 2025 at 10:40
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Union, Any, Dict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .logging_config import get_logger

logger = get_logger("utils")


def validate_file_size(file_path: Path, max_size_mb: float = 100) -> bool:
    """
    Check if a file is within acceptable size limits.
    
    Args:
        file_path: Path to file to check
        max_size_mb: Maximum file size in MB
        
    Returns:
        True if file is acceptable size, False otherwise
    """
    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        return size_mb <= max_size_mb
    except (OSError, FileNotFoundError):
        return False


def create_timestamped_directory(base_dir: Path, prefix: str = "") -> Path:
    """
    Create a directory with timestamp suffix.
    
    Args:
        base_dir: Base directory for new folder
        prefix: Optional prefix for directory name
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{prefix}_{timestamp}" if prefix else timestamp
    new_dir = base_dir / dir_name
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if division by zero
        
    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        x1, y1: Coordinates of first point
        x2, y2: Coordinates of second point
        
    Returns:
        Distance between points
    """
    try:
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    except (TypeError, ValueError):
        return 0.0


def calculate_distance_with_gaps(df):
    """
    Calculate step-wise distances while properly handling missing data gaps.
    
    This function skips over NaN/missing values to avoid incorrect distance calculations
    during periods where no animal is detected (blank frames).
    
    Args:
        df: DataFrame with 'x' and 'y' columns
        
    Returns:
        pandas Series with distance values, NaN for missing data periods
        
    Example:
        Frame  X    Y    -> Distance
        1      200  100  -> NaN (first point)
        2      300  150  -> 111.8 (from frame 1)
        3      NaN  NaN  -> NaN (missing data)
        4      NaN  NaN  -> NaN (missing data)  
        5      400  200  -> 111.8 (from frame 2, skipping gaps)
    """
    import pandas as pd
    import numpy as np
    
    distances = pd.Series(index=df.index, dtype=float)
    distances.iloc[0] = np.nan  # First point has no previous point
    
    last_valid_idx = 0
    last_valid_x = df.iloc[0]['x']
    last_valid_y = df.iloc[0]['y']
    
    for i in range(1, len(df)):
        current_x = df.iloc[i]['x']
        current_y = df.iloc[i]['y']
        
        # Check if current point is valid (not NaN)
        if pd.isna(current_x) or pd.isna(current_y):
            distances.iloc[i] = np.nan
            continue
        
        # Check if we have a valid previous point
        if pd.isna(last_valid_x) or pd.isna(last_valid_y):
            # First valid point after a gap
            distances.iloc[i] = np.nan
        else:
            # Calculate distance from last valid point
            distance = calculate_distance(last_valid_x, last_valid_y, current_x, current_y)
            distances.iloc[i] = distance
        
        # Update last valid point
        last_valid_x = current_x
        last_valid_y = current_y
        last_valid_idx = i
    
    return distances


def get_gap_statistics(df):
    """
    Calculate statistics about missing data gaps in tracking data.
    
    Args:
        df: DataFrame with tracking data
        
    Returns:
        Dictionary with gap statistics
    """
    import pandas as pd
    import numpy as np
    
    # Identify missing points
    missing_mask = pd.isna(df['x']) | pd.isna(df['y'])
    
    if not missing_mask.any():
        return {
            'total_gaps': 0,
            'total_missing_frames': 0,
            'longest_gap': 0,
            'missing_percentage': 0.0,
            'total_gap_duration': 0.0,
            'gap_details': []
        }
    
    # Find gap periods
    gap_starts = []
    gap_ends = []
    in_gap = False
    
    for i, is_missing in enumerate(missing_mask):
        if is_missing and not in_gap:
            # Start of a new gap
            gap_starts.append(i)
            in_gap = True
        elif not is_missing and in_gap:
            # End of current gap
            gap_ends.append(i - 1)
            in_gap = False
    
    # Handle case where data ends with a gap
    if in_gap:
        gap_ends.append(len(missing_mask) - 1)
    
    # Calculate gap statistics
    gap_lengths = [end - start + 1 for start, end in zip(gap_starts, gap_ends)]
    gap_details = []
    total_gap_duration = 0.0
    
    for start, end, length in zip(gap_starts, gap_ends, gap_lengths):
        start_time = df.iloc[start]['time'] if start < len(df) else None
        end_time = df.iloc[end]['time'] if end < len(df) else None
        gap_duration = (end_time - start_time) if (start_time is not None and end_time is not None) else 0.0
        total_gap_duration += gap_duration
        
        gap_details.append({
            'start_frame': start,
            'end_frame': end,
            'length': length,
            'start_time': start_time,
            'end_time': end_time,
            'duration': gap_duration
        })
    
    return {
        'total_gaps': len(gap_starts),
        'total_missing_frames': missing_mask.sum(),
        'longest_gap': max(gap_lengths) if gap_lengths else 0,
        'missing_percentage': (missing_mask.sum() / len(missing_mask)) * 100,
        'total_gap_duration': total_gap_duration,
        'gap_details': gap_details
    }


def smooth_trajectory(
    x_coords: List[float], 
    y_coords: List[float], 
    window_size: int = 3
) -> Tuple[List[float], List[float]]:
    """
    Apply moving average smoothing to trajectory coordinates.
    
    Args:
        x_coords: List of X coordinates
        y_coords: List of Y coordinates
        window_size: Size of smoothing window
        
    Returns:
        Tuple of (smoothed_x, smoothed_y) coordinates
    """
    if len(x_coords) != len(y_coords):
        raise ValueError("X and Y coordinate lists must have same length")
    
    if len(x_coords) < window_size:
        return x_coords.copy(), y_coords.copy()
    
    # Convert to pandas Series for easy rolling mean
    x_series = pd.Series(x_coords)
    y_series = pd.Series(y_coords)
    
    smoothed_x = x_series.rolling(window=window_size, center=True, min_periods=1).mean()
    smoothed_y = y_series.rolling(window=window_size, center=True, min_periods=1).mean()
    
    return smoothed_x.tolist(), smoothed_y.tolist()


def interpolate_missing_points(
    times: List[float],
    x_coords: List[float], 
    y_coords: List[float],
    max_gap: float = 1.0
) -> Tuple[List[float], List[float], List[float]]:
    """
    Interpolate missing points in trajectory data.
    
    Args:
        times: List of time values
        x_coords: List of X coordinates
        y_coords: List of Y coordinates
        max_gap: Maximum time gap to interpolate across (seconds)
        
    Returns:
        Tuple of (interpolated_times, interpolated_x, interpolated_y)
    """
    if len(times) != len(x_coords) or len(times) != len(y_coords):
        raise ValueError("Time, X, and Y lists must have same length")
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        'time': times,
        'x': x_coords,
        'y': y_coords
    })
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    
    # Find gaps larger than max_gap
    df['time_diff'] = df['time'].diff()
    
    # Only interpolate small gaps
    mask = df['time_diff'] <= max_gap
    
    # Interpolate coordinates
    df.loc[mask, 'x'] = df.loc[mask, 'x'].interpolate(method='linear')
    df.loc[mask, 'y'] = df.loc[mask, 'y'].interpolate(method='linear')
    
    return df['time'].tolist(), df['x'].tolist(), df['y'].tolist()


def detect_outliers(
    values: List[float], 
    method: str = "iqr", 
    threshold: float = 1.5
) -> List[bool]:
    """
    Detect outliers in a list of values.
    
    Args:
        values: List of values to analyze
        method: Method to use ("iqr" or "zscore")
        threshold: Threshold for outlier detection
        
    Returns:
        List of boolean values indicating outliers
    """
    if not values:
        return []
    
    values_array = np.array(values)
    
    if method.lower() == "iqr":
        q1 = np.percentile(values_array, 25)
        q3 = np.percentile(values_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (values_array < lower_bound) | (values_array > upper_bound)
    
    elif method.lower() == "zscore":
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        z_scores = np.abs((values_array - mean_val) / std_val) if std_val > 0 else np.zeros_like(values_array)
        outliers = z_scores > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return outliers.tolist()


def format_duration(seconds: float, precision: int = 2) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        precision: Decimal precision for seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 0:
        return "0s"
    
    if seconds < 60:
        return f"{seconds:.{precision}f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.{precision}f}s"
    
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    
    return f"{hours}h {remaining_minutes}m {remaining_seconds:.{precision}f}s"


def format_distance(distance_mm: float, unit: str = "auto") -> str:
    """
    Format distance with appropriate units.
    
    Args:
        distance_mm: Distance in millimeters
        unit: Unit to use ("mm", "cm", "m", "auto")
        
    Returns:
        Formatted distance string
    """
    if distance_mm < 0:
        return "0 mm"
    
    if unit == "auto":
        if distance_mm < 10:
            unit = "mm"
        elif distance_mm < 1000:
            unit = "cm"
        else:
            unit = "m"
    
    if unit == "mm":
        return f"{distance_mm:.1f} mm"
    elif unit == "cm":
        return f"{distance_mm/10:.1f} cm"
    elif unit == "m":
        return f"{distance_mm/1000:.2f} m"
    else:
        return f"{distance_mm:.1f} mm"


def validate_coordinates(x: float, y: float) -> bool:
    """
    Validate that coordinates are finite numbers.
    
    Args:
        x: X coordinate
        y: Y coordinate
        
    Returns:
        True if coordinates are valid, False otherwise
    """
    try:
        return (
            isinstance(x, (int, float)) and 
            isinstance(y, (int, float)) and
            np.isfinite(x) and 
            np.isfinite(y)
        )
    except (TypeError, ValueError):
        return False


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging and logging.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "disk_total_gb": disk.total / (1024**3),
            "disk_free_gb": disk.free / (1024**3),
            "working_directory": str(Path.cwd())
        }
    except Exception as e:
        logger.warning(f"Could not collect system info: {e}")
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "working_directory": str(Path.cwd())
        }


def estimate_memory_usage(n_points: int, n_zones: int) -> float:
    """
    Estimate memory usage for analysis.
    
    Args:
        n_points: Number of data points
        n_zones: Number of zones
        
    Returns:
        Estimated memory usage in MB
    """
    # Rough estimation based on data structures
    # TrackingPoint: ~80 bytes each
    # Zone analysis: ~200 bytes per point per zone
    
    base_memory = n_points * 80 / (1024**2)  # MB
    analysis_memory = n_points * n_zones * 200 / (1024**2)  # MB
    
    return base_memory + analysis_memory


def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are available.
    
    Returns:
        Dictionary with dependency availability status
    """
    dependencies = {
        "pandas": False,
        "numpy": False,
        "tqdm": False,
        "rich": False,
        "typer": False
    }
    
    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies
