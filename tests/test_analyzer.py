"""
Test suite for ToxTrac Data Analyzer.

This module contains unit tests for the core functionality of the analyzer.

Author: Aman Rathore
Contact: amanr.me | amanrathore9753 <at> gmail <dot> com
Created on: Monday, July 14, 2025 at 10:40
"""

import pytest
import sys
from pathlib import Path

# Add src directory to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models import TrackingPoint, BoundingBox, TrackingSession
from src.analyzer import ToxTracAnalyzer
from src.data_loader import ToxTracDataLoader


class TestModels:
    """Test core data models."""
    
    def test_tracking_point_creation(self):
        """Test TrackingPoint creation."""
        point = TrackingPoint(
            time=1.0,
            arena=1,
            track=1,
            x=100.0,
            y=200.0,
            label=1
        )
        
        assert point.time == 1.0
        assert point.x == 100.0
        assert point.y == 200.0
    
    def test_bounding_box_contains_point(self):
        """Test BoundingBox point containment."""
        box = BoundingBox(
            name="test_zone",
            x_min=0.0,
            y_min=0.0,
            x_max=100.0,
            y_max=100.0
        )
        
        # Point inside
        assert box.contains_point(50.0, 50.0) == True
        
        # Point outside
        assert box.contains_point(150.0, 150.0) == False
        
        # Point on boundary
        assert box.contains_point(100.0, 100.0) == True
    
    def test_tracking_session_dataframe(self):
        """Test TrackingSession DataFrame conversion."""
        points = [
            TrackingPoint(time=0.0, arena=1, track=1, x=0.0, y=0.0, label=1),
            TrackingPoint(time=1.0, arena=1, track=1, x=10.0, y=10.0, label=1),
        ]
        
        session = TrackingSession(
            session_name="test",
            data_points=points,
            source_file=Path("test.txt"),
            total_duration=1.0,
            sampling_rate=2.0
        )
        
        df = session.get_dataframe()
        assert len(df) == 2
        assert list(df.columns) == ['time', 'arena', 'track', 'x', 'y', 'label']


class TestAnalyzer:
    """Test analysis functionality."""
    
    def test_zone_analysis(self):
        """Test basic zone analysis."""
        # Create test data
        points = []
        for i in range(100):
            # First 50 points inside zone, next 50 outside
            x = 25.0 if i < 50 else 75.0
            y = 25.0
            points.append(TrackingPoint(
                time=i * 0.1,
                arena=1,
                track=1,
                x=x,
                y=y,
                label=1
            ))
        
        session = TrackingSession(
            session_name="test",
            data_points=points,
            source_file=Path("test.txt"),
            total_duration=9.9,
            sampling_rate=10.0
        )
        
        # Create test zone
        zone = BoundingBox(
            name="test_zone",
            x_min=0.0,
            y_min=0.0,
            x_max=50.0,
            y_max=50.0
        )
        
        # Analyze
        analyzer = ToxTracAnalyzer()
        result = analyzer.analyze_zone_occupancy(session, zone)
        
        # Check results
        assert result.zone_name == "test_zone"
        assert result.percentage_inside == pytest.approx(50.0, abs=1.0)


if __name__ == "__main__":
    pytest.main([__file__])
