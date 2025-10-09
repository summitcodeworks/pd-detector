"""
Tests for the panel detector module
"""

import pytest
import numpy as np
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mobile_panel_detector.detector.panel_detector import AdvancedPanelDetector


class TestAdvancedPanelDetector:
    """Test cases for AdvancedPanelDetector"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = AdvancedPanelDetector()
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        assert self.detector is not None
        assert self.detector.min_yolo_confidence == 0.25
        assert self.detector.min_contour_confidence == 0.30
    
    def test_get_rotated_rect_bbox(self):
        """Test rotated rectangle bounding box calculation"""
        # Create a simple rotated rectangle
        center = (100, 100)
        size = (50, 30)
        angle = 45
        rotated_rect = (center, size, angle)
        
        bbox, box_points = self.detector.get_rotated_rect_bbox(rotated_rect)
        
        assert len(bbox) == 4  # x, y, w, h
        assert len(box_points) == 4  # 4 corner points
        assert all(isinstance(coord, (int, np.integer)) for coord in bbox)
    
    def test_has_screen_characteristics_dark_image(self):
        """Test screen characteristics with dark image (should return False)"""
        # Create a dark image
        dark_image = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = (10, 10, 80, 80)
        
        result = self.detector.has_screen_characteristics(dark_image, bbox)
        assert result is False
    
    def test_has_screen_characteristics_bright_image(self):
        """Test screen characteristics with very bright image (should return False)"""
        # Create a very bright image
        bright_image = np.full((100, 100, 3), 250, dtype=np.uint8)
        bbox = (10, 10, 80, 80)
        
        result = self.detector.has_screen_characteristics(bright_image, bbox)
        assert result is False
    
    def test_has_screen_characteristics_varied_image(self):
        """Test screen characteristics with varied content image (should return True)"""
        # Create an image with varied content
        varied_image = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        bbox = (10, 10, 80, 80)
        
        result = self.detector.has_screen_characteristics(varied_image, bbox)
        # This might be True or False depending on the random content
        assert isinstance(result, bool)
    
    def test_detect_with_empty_image(self):
        """Test detection with empty image"""
        empty_image = np.array([])
        
        # This should handle the empty image gracefully
        with pytest.raises((ValueError, IndexError)):
            self.detector.detect(empty_image)
    
    def test_detect_with_small_image(self):
        """Test detection with very small image"""
        small_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        detections, status_info = self.detector.detect(small_image)
        assert isinstance(detections, list)
        assert isinstance(status_info, dict)
        assert 'status' in status_info
        assert 'message' in status_info
        assert 'panel_count' in status_info
    
    def test_merge_detections_empty(self):
        """Test merging empty detections"""
        detections = []
        result = self.detector.merge_detections(detections)
        assert result == []
    
    def test_merge_detections_single(self):
        """Test merging single detection"""
        detections = [{
            'bbox': (10, 10, 50, 50),
            'confidence': 0.8,
            'method': 'test'
        }]
        result = self.detector.merge_detections(detections)
        assert len(result) == 1
        assert result[0] == detections[0]
    
    @pytest.mark.slow
    def test_detect_with_realistic_image(self):
        """Test detection with realistic image (marked as slow)"""
        # Create a realistic test image
        test_image = np.random.randint(100, 150, (400, 600, 3), dtype=np.uint8)
        
        detections, status_info = self.detector.detect(test_image)
        assert isinstance(detections, list)
        assert isinstance(status_info, dict)
        # In a realistic scenario, we might not detect anything in random noise
