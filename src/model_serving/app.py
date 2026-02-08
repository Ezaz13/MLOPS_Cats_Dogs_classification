import sys
import os
import torch
import mlflow.pytorch
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import transforms
import logging

# ------------------------------------------------------------------
# SETUP PATHS
# ------------------------------------------------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
sys.path.append(str(project_root))

from src.utility.logger import setup_logging

# ------------------------------------------------------------------
# FLASK APP
# ------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")

# Disable Flask's default logger to prevent conflicts
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
MLFLOW_DB_PATH = project_root / "mlflow.db"
MLFLOW_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MODEL_NAME = "CatsDogsCNN"

# Class names from ImageFolder (lowercase in dataset folders)
# Model outputs: 0=cat, 1=dog (alphabetical order)
CLASSES = ['cat', 'dog']

# Global variables
device = None
model = None
logger = None

# ------------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------
def initialize_app():
    """Initialize logger, device, and model"""
    global device, logger

    # Force unbuffered output for Windows console visibility
    os.environ['PYTHONUNBUFFERED'] = '1'

    # Print startup banner BEFORE logger setup - ensures immediate visibility
    print("\n" + "=" * 60, flush=True)
    print("CATS vs DOGS - MODEL SERVING API", flush=True)
    print("=" * 60, flush=True)

    # Setup logger
    logger = setup_logging("model_serving")

    logger.info(f"Project Root: {project_root}")
    logger.info(f"MLflow URI: {MLFLOW_URI}")
    logger.info(f"Model Name: {MODEL_NAME}")

    # Setup device - Force CPU since we're loading GPU-trained models with CPU mapping
    # This ensures compatibility when loading models trained on GPU in CPU-only environments
    device = torch.device("cpu")
    logger.info(f"Device: {device} (forced for GPU-trained model compatibility)")
    print("=" * 60 + "\n", flush=True)

    # Load model
    load_model()


def load_model():
    """Load the trained model from MLflow registry"""
    global model
    try:
        # Save original torch.load to restore later
        _original_torch_load = torch.load
        
        # Override torch.load globally to force CPU mapping
        def _force_cpu_load(*args, **kwargs):
            """Force all torch.load calls to use CPU mapping"""
            kwargs['map_location'] = torch.device('cpu')
            return _original_torch_load(*args, **kwargs)
        
        # Apply the override
        torch.load = _force_cpu_load
        
        try:
            model_loaded = False
            
            # Try 1: Check for local model artifact (Docker/Production)
            local_model_path = Path("/app/model")
            if local_model_path.exists():
                logger.info(f"Found Docker model at {local_model_path}")
                logger.info("Loading model with CPU mapping (GPU->CPU compatibility)")
                model = mlflow.pytorch.load_model(f"file://{local_model_path}")
                model_loaded = True
            
            # Try 2: Check for exported model in project (Local Development)
            if not model_loaded:

                export_model_path = project_root / "models" / "model_export"
                if export_model_path.exists():
                    logger.info(f"Found exported model at {export_model_path}")
                    logger.info("Loading model with CPU mapping (GPU->CPU compatibility)")
                    model = mlflow.pytorch.load_model(f"file://{export_model_path.as_posix()}")
                    model_loaded = True
            
            # Try 3: Fallback to MLflow Registry (Local Development)
            if not model_loaded:
                logger.info("Connecting to MLflow tracking server...")
                mlflow.set_tracking_uri(MLFLOW_URI)

                model_uri = f"models:/{MODEL_NAME}/Latest"
                logger.info(f"Loading model from: {model_uri}")
                logger.info("Loading model with CPU mapping (GPU->CPU compatibility)")

                model = mlflow.pytorch.load_model(model_uri)
                model_loaded = True
        finally:
            # Always restore original torch.load
            torch.load = _original_torch_load

        model.to(device)
        model.eval()

        logger.info("✅ Model loaded successfully!")
        logger.info(f"Model is running on: {device}")
        sys.stdout.flush()

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.warning("Make sure you have:")
        logger.warning("  1. Run 'train_model.py' to train the model")
        logger.warning(f"  2. Registered the model as '{MODEL_NAME}' in MLflow")
        sys.stdout.flush()


# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    """Render the web UI"""
    logger.info("Web UI accessed")
    sys.stdout.flush()
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if model is None:
        logger.error("Prediction request received but model not loaded")
        sys.stdout.flush()
        return jsonify({"error": "Model not loaded. Check server logs."}), 503

    if 'file' not in request.files:
        logger.warning("Prediction request missing file")
        sys.stdout.flush()
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("Prediction request with empty filename")
        sys.stdout.flush()
        return jsonify({"error": "No selected file"}), 400

    try:
        logger.info(f"Processing prediction for: {file.filename}")

        # Load and preprocess image
        image = Image.open(file).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)

            # Get predicted class (lowercase from model)
            predicted_class = CLASSES[pred_idx.item()]
            conf_score = confidence.item()

            # Capitalize for display
            display_class = predicted_class.capitalize()

        logger.info(f"✅ Prediction: {display_class} (confidence: {conf_score:.2%})")
        sys.stdout.flush()

        return jsonify({
            "class": display_class,  # Capitalized for display
            "confidence": float(conf_score)
        })

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        sys.stdout.flush()
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }
    return jsonify(status), 200


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == '__main__':
    # Initialize app (logger, device, model)
    initialize_app()

    logger.info("\n" + "=" * 60)
    logger.info("Starting Flask server on http://localhost:5000")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 60 + "\n")
    sys.stdout.flush()

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)