# Material Stream Identification System

**Real-time image-based waste material classifier and sorting pipeline achieving 90% classification accuracy.**

* **Core Architecture:** Features ResNet50 as a CNN feature extractor coupled with a scikit-learn Support Vector Classifier (SVC) pipeline.
* **Data Balancing:** Utilized targeted data augmentation techniques to expand underrepresented classes, effectively mitigating dataset imbalance synthetically.
* **Deployment Utilities:** Supports both live, real-time camera inference and bulk batch prediction capabilities.

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
