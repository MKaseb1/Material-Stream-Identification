# Material Stream Identification System

image-based waste material classifier using a CNN feature extractor (ResNet50) + scikit-learn pipeline. Includes real-time camera prediction and batch prediction utilities.

## Quick start

1. Create a virtual environment (Python 3.8+)

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Place your trained pipeline at `models/svc_pipeline.pkl`.

3. Run the camera demo (uses webcam index 0):

```bash
python src/camera_app.py
```

4. Run batch prediction on a folder of images:

```bash
python src/test.py
# or import `predict()` from `src/test.py` in your scripts
```

## Repository layout

- `src/` - application scripts (`camera_app.py`, `test.py`)
- `models/` - trained model artifacts (ignore large files if necessary)
- `dataset/` - optional image datasets (not required to publish)
- `notebooks/` - exploration and experiments

## Notes

- The code uses TensorFlow / Keras ResNet50 to extract features and a scikit-learn pipeline (saved with `joblib`) for classification. Ensure `models/svc_pipeline.pkl` exists before running the demo.
