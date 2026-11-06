# Synthetic Data Image Classifier — Interactive App

This is a Streamlit web interface for the CIFAR-10 synthetic data augmentation project.

## Run on Windows

1. Install Python 3.10 or 3.11 (recommended).
2. Extract this ZIP folder.
3. Open Command Prompt in the extracted folder.
4. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Start the app:

   ```bash
   streamlit run app.py
   ```

7. Your browser should open the app. If it does not, open the Local URL printed in Command Prompt (usually http://localhost:8501).

## How to use

- Open **Train & Compare** and click **Train models and evaluate**.
- The app downloads CIFAR-10 and trains the baseline and augmented CNNs.
- It displays test accuracy, loss, training time, learning curves and confusion matrices.
- Open **Try an Image** to upload a PNG/JPG and get a predicted class and confidence.
- Trained models are saved in `saved_models/` so you do not need to train again after they have been saved.

## Notes

- The first run needs an internet connection to download CIFAR-10 and may take several minutes.
- A GPU is helpful but not required. On CPU, training may take longer.
- CIFAR-10 has ten classes and 32×32 images. Predictions on unrelated real-world photos may be unreliable.
- This app uses conventional synthetic augmentation (flip, rotation, zoom and translation), not GAN/diffusion-generated images.
