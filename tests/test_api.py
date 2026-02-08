import pytest
import io
import torch
import json
from unittest.mock import patch, MagicMock
from PIL import Image
from src.model_serving.app import app

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def create_dummy_image(color=(255, 0, 0)):
    """Creates a simple in-memory image for testing."""
    img = Image.new('RGB', (224, 224), color=color)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
class TestModelServingAPI:

    @patch('src.model_serving.app.logger')
    @patch('src.model_serving.app.model')
    @patch('src.model_serving.app.device', torch.device('cpu'))
    def test_health_endpoint(self, mock_model, mock_logger, client):
        """Test the /health endpoint with mocked model."""
        
        # Test when model is loaded (mocked)
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "device" in data

    @patch('src.model_serving.app.logger')
    def test_health_endpoint_no_model(self, mock_logger, client):
        """Test the /health endpoint when model is NOT loaded."""
        # We need to explicitly set app.model to None for this test
        # because the app might have tried to load it on import
        with patch('src.model_serving.app.model', None):
            response = client.get('/health')
            assert response.status_code == 200
            data = response.json
            assert data["model_loaded"] is False

    @pytest.mark.parametrize("color, expected_class", [
        ((255, 0, 0), "Cat"),   # Dummy color mapping
        ((0, 255, 0), "Dog"),   # Dummy color mapping
    ])
    @patch('src.model_serving.app.logger')
    @patch('src.model_serving.app.model')
    @patch('src.model_serving.app.device', torch.device('cpu'))
    def test_prediction_endpoint(self, mock_model, mock_logger, color, expected_class, client):
        """Test the /predict endpoint with mocked model predictions."""
        
        # Setup mock model output
        # Output shape: [1, 2] (batch_size, num_classes)
        # Class 0 = Cat, Class 1 = Dog
        if expected_class == "Cat":
            # Higher score for index 0
            mock_output = torch.tensor([[5.0, -5.0]])
        else:
            # Higher score for index 1
            mock_output = torch.tensor([[-5.0, 5.0]])
            
        mock_model.return_value = mock_output

        # Create dummy image
        img_bytes = create_dummy_image(color)
        data = {'file': (img_bytes, 'test.jpg')}

        # Make request
        response = client.post('/predict', data=data, content_type='multipart/form-data')
        
        assert response.status_code == 200
        json_response = response.json
        
        assert "class" in json_response
        assert "confidence" in json_response
        assert json_response["class"] == expected_class

    @patch('src.model_serving.app.logger')
    @patch('src.model_serving.app.model')
    def test_prediction_no_file(self, mock_model, mock_logger, client):
        """Test prediction without file."""
        response = client.post('/predict', data={})
        assert response.status_code == 400
        assert "error" in response.json
