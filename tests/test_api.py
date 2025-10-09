"""
Tests for the API routes
"""

import pytest
import json
import sys
import os
from io import BytesIO

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mobile_panel_detector.app import create_app


class TestAPI:
    """Test cases for API endpoints"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.app = create_app('testing')
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'service' in data
        assert 'version' in data
        assert 'detection_mode' in data
    
    def test_detect_endpoint_no_file(self):
        """Test detect endpoint with no file"""
        response = self.client.post('/detect')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No image provided' in data['error']
    
    def test_detect_endpoint_empty_file(self):
        """Test detect endpoint with empty file"""
        response = self.client.post('/detect', 
                                  data={'file': (BytesIO(b''), '')})
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No file selected' in data['error']
    
    def test_detect_endpoint_invalid_json(self):
        """Test detect endpoint with invalid JSON"""
        response = self.client.post('/detect',
                                  data=json.dumps({'invalid': 'data'}),
                                  content_type='application/json')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No image provided' in data['error']
    
    def test_detect_image_endpoint_no_file(self):
        """Test detect/image endpoint with no file"""
        response = self.client.post('/detect/image')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No file provided' in data['error']
    
    def test_detect_batch_endpoint_no_files(self):
        """Test detect/batch endpoint with no files"""
        response = self.client.post('/detect/batch')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No files provided' in data['error']
    
    def test_processed_image_endpoint_not_found(self):
        """Test processed image endpoint with non-existent file"""
        response = self.client.get('/processed/nonexistent.jpg')
        assert response.status_code == 404
    
    @pytest.mark.integration
    def test_full_detection_workflow(self):
        """Test full detection workflow with mock image"""
        # Create a simple test image
        import numpy as np
        import cv2
        
        test_image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', test_image)
        
        response = self.client.post('/detect',
                                  data={'file': (BytesIO(buffer), 'test.jpg')})
        
        # Should return 200 even if no panels detected
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'detected' in data
        assert 'result' in data
        assert 'panel_count' in data
        assert 'confidence' in data
        assert 'panels' in data
        assert 'display_issues' in data
        assert 'has_display_issues' in data
        assert 'display_issue_count' in data
        assert 'issue_summary' in data
        assert 'processed_image_url' in data
        assert 'saved_filename' in data
        assert 'message' in data
        assert 'image_size' in data
        assert 'detection_mode' in data
