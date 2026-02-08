import pytest
import requests
import io
import time
from PIL import Image

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
API_URL = "http://127.0.0.1:5000/predict"

def create_dummy_image(color=(255, 0, 0)):
    """Creates a simple in-memory image for testing."""
    img = Image.new('RGB', (224, 224), color=color)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

class TestModelServingAPI:
    def test_api_connectivity(self):
        """Test if the API endpoint is reachable."""
        try:
            # Just check if we can connect (even if GET isn't allowed, connection should work)
            # Actually URL is /predict which expects POST.
            # Let's try to connect to root / to see if server is up
            root_url = API_URL.replace("/predict", "/")
            response = requests.get(root_url)
            assert response.status_code == 200, "Server root accessible"
        except requests.exceptions.ConnectionError:
            pytest.fail(f"Could not connect to API at {API_URL}. Ensure server is running on port 5000.")

    def test_health_endpoint(self):
        """Test the /health endpoint."""
        health_url = API_URL.replace("/predict", "/health")
        try:
            response = requests.get(health_url)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "model_loaded" in data
            assert "device" in data
        except requests.exceptions.ConnectionError:
            pytest.fail("Connection refused. Is the Flask app running?")

    @pytest.mark.parametrize("color, expected_status", [
        ((255, 0, 0), 200),   # Red
        ((0, 255, 0), 200),   # Green
        ((0, 0, 255), 200),   # Blue
    ])
    def test_prediction_endpoint(self, color, expected_status):
        """Test the /predict endpoint with valid images."""
        img_bytes = create_dummy_image(color)
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        
        try:
            response = requests.post(API_URL, files=files)
            assert response.status_code == expected_status
            
            # Check JSON structure
            json_response = response.json()
            assert "class" in json_response
            assert "confidence" in json_response
            assert json_response["class"] in ["Cat", "Dog"]
            assert 0.0 <= json_response["confidence"] <= 1.0
            
        except requests.exceptions.ConnectionError:
            pytest.fail("Connection refused. Is the Flask app running?")
