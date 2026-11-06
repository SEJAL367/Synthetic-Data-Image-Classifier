import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

st.set_page_config(
    page_title="Synthetic Data Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)
BASELINE_PATH = MODEL_DIR / "baseline_cnn.keras"
AUGMENTED_PATH = MODEL_DIR / "augmented_cnn.keras"

st.title("🖼️ Synthetic Data Image Classifier")
st.caption("Compare a CNN trained on original CIFAR-10 images with a CNN trained using synthetic data augmentation.")

with st.sidebar:
    st.header("Project settings")
    epochs = st.slider("Training epochs", min_value=1, max_value=20, value=5)
    batch_size = st.selectbox("Batch size", [32, 64, 128], index=1)
    st.info("First training downloads CIFAR-10 automatically. Training time depends on your computer and whether a GPU is available.")
    st.caption(f"TensorFlow: {tf.__version__}")
    st.caption("GPU available: " + ("Yes" if tf.config.list_physical_devices("GPU") else "No"))

@st.cache_data(show_spinner=False)
def load_data():
    (x_train_all, y_train_all), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train_all = y_train_all.squeeze()
    y_test = y_test.squeeze()
    rng = np.random.default_rng(42)
    indices = np.arange(len(x_train_all))
    rng.shuffle(indices)
    x_train_all = x_train_all[indices]
    y_train_all = y_train_all[indices]
    val_size = 5000
    x_val = x_train_all[:val_size].astype("float32") / 255.0
    y_val = y_train_all[:val_size]
    x_train = x_train_all[val_size:].astype("float32") / 255.0
    y_train = y_train_all[val_size:]
    x_test = x_test.astype("float32") / 255.0
    return x_train, y_train, x_val, y_val, x_test, y_test

def build_cnn(use_augmentation=False):
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.08, 0.08)
    ], name="data_augmentation")

    inputs = keras.Input(shape=(32, 32, 3))
    x = inputs
    if use_augmentation:
        x = augmentation(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_one(model, x_train, y_train, x_val, y_val, epochs, batch_size, label):
    progress = st.progress(0, text=f"Training {label}...")
    status = st.empty()
    class ProgressCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress.progress((epoch + 1) / epochs,
                              text=f"{label}: completed epoch {epoch+1}/{epochs}")
            logs = logs or {}
            status.write(
                f"Epoch {epoch+1}/{epochs} — "
                f"training accuracy: {logs.get('accuracy', 0):.3f}, "
                f"validation accuracy: {logs.get('val_accuracy', 0):.3f}"
            )
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            ProgressCallback(),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],
        verbose=0
    )
    progress.progress(1.0, text=f"{label} training complete")
    return history

def load_saved_models():
    models = {}
    if BASELINE_PATH.exists():
        models["Baseline CNN"] = keras.models.load_model(BASELINE_PATH)
    if AUGMENTED_PATH.exists():
        models["CNN + Augmentation"] = keras.models.load_model(AUGMENTED_PATH)
    return models

tab_overview, tab_train, tab_predict, tab_about = st.tabs(
    ["Overview", "Train & Compare", "Try an Image", "About"]
)

with tab_overview:
    st.subheader("Project overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset", "CIFAR-10")
    c2.metric("Image classes", "10")
    c3.metric("Image size", "32 × 32")
    st.write(
        "This project tests whether synthetic image augmentation helps a CNN "
        "generalize better. The same CNN architecture is trained in two ways: "
        "using original training images, and using randomly augmented training images."
    )
    st.markdown("**Augmentation techniques used**")
    st.write("Horizontal flipping · Small rotation · Zoom · Translation")
    st.warning(
        "This version uses conventional synthetic data augmentation. "
        "It does not generate new images with a GAN or diffusion model."
    )

with tab_train:
    st.subheader("Train both models")
    st.write("Press the button below to download the dataset, train both models and compare their test performance.")
    if st.button("🚀 Train models and evaluate", type="primary", use_container_width=True):
        try:
            with st.spinner("Loading CIFAR-10 dataset..."):
                x_train, y_train, x_val, y_val, x_test, y_test = load_data()
            st.success(
                f"Dataset ready: {len(x_train):,} training, "
                f"{len(x_val):,} validation and {len(x_test):,} test images."
            )

            start = time.time()
            baseline = build_cnn(False)
            baseline_history = train_one(
                baseline, x_train, y_train, x_val, y_val,
                epochs, batch_size, "Baseline CNN"
            )
            baseline_time = time.time() - start
            baseline.save(BASELINE_PATH)

            start = time.time()
            augmented = build_cnn(True)
            augmented_history = train_one(
                augmented, x_train, y_train, x_val, y_val,
                epochs, batch_size, "CNN with augmentation"
            )
            augmented_time = time.time() - start
            augmented.save(AUGMENTED_PATH)

            with st.spinner("Evaluating models on the untouched test set..."):
                base_loss, base_acc = baseline.evaluate(x_test, y_test, verbose=0)
                aug_loss, aug_acc = augmented.evaluate(x_test, y_test, verbose=0)
                base_pred = np.argmax(baseline.predict(x_test, verbose=0), axis=1)
                aug_pred = np.argmax(augmented.predict(x_test, verbose=0), axis=1)

            st.session_state["results"] = {
                "base_acc": float(base_acc), "aug_acc": float(aug_acc),
                "base_loss": float(base_loss), "aug_loss": float(aug_loss),
                "base_time": float(baseline_time), "aug_time": float(augmented_time),
                "base_history": {k: [float(v) for v in vals] for k, vals in baseline_history.history.items()},
                "aug_history": {k: [float(v) for v in vals] for k, vals in augmented_history.history.items()},
                "base_pred": base_pred, "aug_pred": aug_pred, "y_test": y_test
            }
            st.success("Both models trained and saved successfully.")
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.exception(e)

    results = st.session_state.get("results")
    if results:
        st.markdown("### Test-set results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline accuracy", f"{results['base_acc']*100:.2f}%")
        m2.metric("Augmented accuracy", f"{results['aug_acc']*100:.2f}%")
        m3.metric("Difference", f"{(results['aug_acc']-results['base_acc'])*100:+.2f} pp")
        st.caption("pp = percentage points. These are measured results from the current run.")
        st.dataframe({
            "Model": ["Baseline CNN", "CNN + Augmentation"],
            "Test accuracy (%)": [results["base_acc"]*100, results["aug_acc"]*100],
            "Test loss": [results["base_loss"], results["aug_loss"]],
            "Training time (min)": [results["base_time"]/60, results["aug_time"]/60]
        }, use_container_width=True)

        st.markdown("### Accuracy curves")
        fig, ax = plt.subplots(figsize=(9, 4))
        for key, label in [("base_history", "Baseline"), ("aug_history", "Augmented")]:
            h = results[key]
            ax.plot(h["accuracy"], label=f"{label} train")
            ax.plot(h["val_accuracy"], linestyle="--", label=f"{label} validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Loss curves")
        fig, ax = plt.subplots(figsize=(9, 4))
        for key, label in [("base_history", "Baseline"), ("aug_history", "Augmented")]:
            h = results[key]
            ax.plot(h["loss"], label=f"{label} train")
            ax.plot(h["val_loss"], linestyle="--", label=f"{label} validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Confusion matrices")
        cm1, cm2 = st.columns(2)
        for col, preds, title in [
            (cm1, results["base_pred"], "Baseline CNN"),
            (cm2, results["aug_pred"], "CNN + Augmentation")
        ]:
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(
                confusion_matrix(results["y_test"], preds),
                display_labels=CLASS_NAMES
            )
            disp.plot(ax=ax, xticks_rotation=45, colorbar=False, values_format="d")
            ax.set_title(title)
            col.pyplot(fig)
            plt.close(fig)

with tab_predict:
    st.subheader("Classify your own image")
    models = load_saved_models()
    if not models:
        st.info("Train the models first in the 'Train & Compare' tab. The trained models will be saved locally.")
    else:
        selected_model_name = st.selectbox("Choose model", list(models.keys()))
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            left, right = st.columns([1, 1])
            with left:
                st.image(image, caption="Uploaded image", use_container_width=True)
            resized = image.resize((32, 32))
            arr = np.asarray(resized).astype("float32") / 255.0
            prediction = models[selected_model_name].predict(np.expand_dims(arr, 0), verbose=0)[0]
            index = int(np.argmax(prediction))
            with right:
                st.markdown("### Prediction")
                st.success(f"**{CLASS_NAMES[index].title()}**")
                st.metric("Model confidence", f"{prediction[index]*100:.2f}%")
                st.markdown("**Class probabilities**")
                for i in np.argsort(prediction)[::-1]:
                    st.write(f"{CLASS_NAMES[i].title()}: {prediction[i]*100:.1f}%")
                    st.progress(float(prediction[i]))
            st.caption("CIFAR-10 contains small 32×32 images. Predictions on unrelated, high-resolution photos may be unreliable.")

with tab_about:
    st.subheader("How the project works")
    st.markdown("""
    1. **Dataset:** CIFAR-10 provides labelled images across ten categories.
    2. **Preprocessing:** Pixel values are normalized to the 0–1 range.
    3. **Baseline model:** A CNN learns from original training images.
    4. **Augmented model:** The same CNN learns while random transformations are applied to training images.
    5. **Evaluation:** Both models are tested on the same untouched test set.
    6. **Comparison:** Accuracy, loss, learning curves and confusion matrices are displayed.

    **Subject integration**
    - **AI:** automated image recognition and prediction.
    - **ML:** supervised learning, training/testing and generalization.
    - **DL:** CNN-based automatic feature extraction and classification.

    **Important distinction:** Rotation, flipping, zoom and translation are data augmentation methods, not GAN/diffusion-based Generative AI.
    """)
    st.markdown("**Suggested project title:** Image Classification Using Synthetic Data Augmentation with Convolutional Neural Networks.")
