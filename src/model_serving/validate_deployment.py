import sys
import requests
import time
import io
from pathlib import Path
from PIL import Image

# ------------------------------------------------------------------
# Setup Paths & Logging
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utility.logger import setup_logging
logger = setup_logging("deployment_validation")

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
API_URL = "http://127.0.0.1:5001/predict"


def create_dummy_image(color=(255, 0, 0)):
    """Creates a simple in-memory image for testing."""
    img = Image.new('RGB', (224, 224), color=color)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

def run_tests():
    logger.info("=" * 70)
    logger.info("🧪 TESTING MODEL SERVING DEPLOYMENT")
    logger.info("=" * 70)
    logger.info(f"Target API: {API_URL}")
    logger.info("-" * 70)
    logger.info(f"{'Test Case':<20} | {'Prediction':<15} | {'Confidence':<12} | {'Status':<10}")
    logger.info("-" * 70)

    # Test cases: Sending random colored images to ensure the pipeline works
    test_cases = [
        ("Red Image", (255, 0, 0)),
        ("Green Image", (0, 255, 0)),
        ("Blue Image", (0, 0, 255)),
        ("Yellow Image", (255, 255, 0)),
        ("White Image", (255, 255, 255))
    ]

    success_count = 0
    failed_count = 0

    for name, color in test_cases:
        try:
            img_bytes = create_dummy_image(color)
            files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
            
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                result = response.json()
                pred = result.get('class', 'N/A')
                conf = result.get('confidence', 0.0)
                status = "OK"
                logger.info(f"{name:<20} | {pred:<15} | {conf*100:>6.2f}%      | {status:<10}")
                success_count += 1
            else:
                logger.error(f"{name:<20} | {'Error':<15} | {'N/A':<12} | HTTP {response.status_code}")
                failed_count += 1
            
            # Small delay between requests
            time.sleep(0.2)
            
        except requests.exceptions.ConnectionError:
            logger.error("=" * 70)
            logger.error("CONNECTION ERROR")
            logger.error("=" * 70)
            logger.error(f"Cannot connect to the API at {API_URL}")
            logger.error("Make sure the Flask app is running:")
            logger.error("  python src/model_serving/app.py")
            logger.error("=" * 70)
            return
        except Exception as e:
            logger.error(f"{name:<20} | Error: {str(e)}")
            failed_count += 1

    # Summary
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total Tests:  {len(test_cases)}")
    logger.info(f"Passed:    {success_count}")
    logger.info(f"Failed:    {failed_count}")
    logger.info("=" * 70)
    
    if success_count == len(test_cases):
        logger.info("🎉 All tests passed! Deployment is working correctly.")
    elif success_count > 0:
        logger.warning("⚠️  Some tests failed. Check the errors above.")
    else:
        logger.error("❌ All tests failed. There may be an issue with the deployment.")
    
    logger.info("=" * 70)

if __name__ == "__main__":
    logger.info("Starting deployment validation tests...")
    run_tests()