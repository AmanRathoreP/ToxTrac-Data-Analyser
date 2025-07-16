"""
Configuration settings for ToxTrac Data Analyzer.

Author: Aman Rathore
Contact: amanr.me | amanrathore9753 <at> gmail <dot> com
Created on: Monday, July 14, 2025 at 10:40
"""

from pathlib import Path
from typing import Dict, Any, List

# Application metadata
APP_NAME = "ToxTrac Data Analyzer"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Aman Rathore"
APP_EMAIL = "amanrathore9753@gmail.com"

# Default file patterns for ToxTrac data
TRACKING_FILE_PATTERNS = [
    "**/Tracking_RealSpace*.txt",
    "**/Tracking_*.txt"
]

# Supported analysis types
ZONE_TYPES = ["epm", "open-field", "oft", "oft-individual", "custom"]
OUTPUT_FORMATS = ["csv", "json", "console"]

# Default zone configurations
DEFAULT_EPM_ZONES = {
    "center_size_ratio": 0.2,  # Center zone is 20% of total area
    "zones": ["center", "open_arm_east", "open_arm_west", "closed_arm_north", "closed_arm_south"]
}

DEFAULT_OPEN_FIELD_ZONES = {
    "center_ratio": 0.33,      # Center zone is inner 33%
    "middle_ratio": 0.66,      # Middle zone extends to 66%
    "zones": ["center", "middle", "peripheral"]
}

DEFAULT_OFT_ZONES = {
    "inner_zone_ratio": 0.6,   # Inner zone is 60% of outer zone size
    "zones": ["inner", "outer"]
}

# Analysis settings
DEFAULT_ANALYSIS_SETTINGS = {
    "calculate_movement": True,
    "minimum_visit_duration": 0.0,  # Minimum time to count as a visit (seconds)
    "interpolate_missing_points": True,  # Enable linear interpolation by default
    "smooth_trajectory": False
}

# Output settings
DEFAULT_OUTPUT_SETTINGS = {
    "decimal_places": 2,
    "include_summary_stats": True,
    "create_timestamped_folders": True,
    "compress_output": False
}

# Logging settings
DEFAULT_LOG_SETTINGS = {
    "level": "INFO",
    "format": "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    "rich_tracebacks": True,
    "show_path": True
}

# File size limits (for safety)
MAX_FILE_SIZE_MB = 100
MAX_DATA_POINTS = 1000000

# Zone validation settings
ZONE_VALIDATION = {
    "allow_overlapping_zones": True,
    "require_complete_coverage": False,
    "minimum_zone_size": 1.0  # Minimum zone area in mm²
}

class Config:
    """Configuration class for the application."""
    
    def __init__(self):
        """Initialize configuration with defaults."""
        self.app_name = APP_NAME
        self.app_version = APP_VERSION
        self.app_author = APP_AUTHOR
        self.app_email = APP_EMAIL
        
        self.tracking_patterns = TRACKING_FILE_PATTERNS.copy()
        self.zone_types = ZONE_TYPES.copy()
        self.output_formats = OUTPUT_FORMATS.copy()
        
        self.epm_zones = DEFAULT_EPM_ZONES.copy()
        self.open_field_zones = DEFAULT_OPEN_FIELD_ZONES.copy()
        self.oft_zones = DEFAULT_OFT_ZONES.copy()
        
        self.analysis_settings = DEFAULT_ANALYSIS_SETTINGS.copy()
        self.output_settings = DEFAULT_OUTPUT_SETTINGS.copy()
        self.log_settings = DEFAULT_LOG_SETTINGS.copy()
        
        self.max_file_size_mb = MAX_FILE_SIZE_MB
        self.max_data_points = MAX_DATA_POINTS
        
        self.zone_validation = ZONE_VALIDATION.copy()
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "app_author": self.app_author,
            "app_email": self.app_email,
            "tracking_patterns": self.tracking_patterns,
            "zone_types": self.zone_types,
            "output_formats": self.output_formats,
            "epm_zones": self.epm_zones,
            "open_field_zones": self.open_field_zones,
            "oft_zones": self.oft_zones,
            "analysis_settings": self.analysis_settings,
            "output_settings": self.output_settings,
            "log_settings": self.log_settings,
            "max_file_size_mb": self.max_file_size_mb,
            "max_data_points": self.max_data_points,
            "zone_validation": self.zone_validation
        }

# Global configuration instance
config = Config()
