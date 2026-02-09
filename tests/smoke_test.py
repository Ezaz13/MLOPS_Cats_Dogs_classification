import requests
import time
import sys
import os
import io
from PIL import Image

# Configuration
SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:5000")
MAX_RETRIES = 5
RETRY_DELAY = 5

def check_health():
    """Check the /health endpoint"""
    url = f"{SERVICE_URL}/health"
    print(f"Checking health at {url}...")
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy" and data.get("model_loaded") is True:
                print("✅ Health check PASSED")
                return True
            else:
                print(f"❌ Health check FAILED: {data}")
        else:
            print(f"❌ Health check FAILED with status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check ERROR: {e}")
    
    return False

def check_prediction():
    """Check the /predict endpoint with a dummy image"""
    url = f"{SERVICE_URL}/predict"
    print(f"Checking prediction at {url}...")

    # Create a dummy white image
    img = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    files = {'file': ('dummy.jpg', img_byte_arr, 'image/jpeg')}

    try:
        response = requests.post(url, files=files, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "class" in data and "confidence" in data:
                print(f"✅ Prediction check PASSED. Result: {data}")
                return True
            else:
                print(f"❌ Prediction check FAILED: Invalid response format {data}")
        else:
            print(f"❌ Prediction check FAILED with status code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Prediction check ERROR: {e}")

    return False

def main():
    print(f"Starting smoke tests against {SERVICE_URL}...")
    
    # 1. Health Check with Retries
    health_passed = False
    for i in range(MAX_RETRIES):
        if check_health():
            health_passed = True
            break
        print(f"Retrying health check in {RETRY_DELAY} seconds... ({i+1}/{MAX_RETRIES})")
        time.sleep(RETRY_DELAY)

    if not health_passed:
        print("🚨 Smoke tests FAILED: Service is not healthy.")
        sys.exit(1)

    # 2. Prediction Check
    if check_prediction():
        print("🎉 All smoke tests PASSED!")
        sys.exit(0)
    else:
        print("🚨 Smoke tests FAILED: Prediction endpoint failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
