# Project Report — Interactive Synthetic Data Image Classifier

## Title
Image Classification Using Synthetic Data Augmentation with Convolutional Neural Networks

## Abstract
This project implements an interactive image classification application using a Convolutional Neural Network (CNN). It investigates whether synthetic data augmentation improves image classification performance. Two CNN models are trained on CIFAR-10: a baseline model trained on original images and an augmented model trained with random flipping, rotation, zoom and translation. Both models are evaluated on the same untouched test set. The application displays performance metrics, learning curves, confusion matrices and predictions for uploaded images.

## Objectives
- Build a CNN classifier for the CIFAR-10 dataset.
- Apply synthetic augmentation to increase training-image diversity.
- Compare a baseline CNN with an augmented CNN.
- Evaluate both models using test accuracy, test loss and confusion matrices.
- Provide an interactive interface for training and image prediction.

## Methodology
1. Load CIFAR-10.
2. Normalize images and reserve a validation subset.
3. Train a baseline CNN on original training images.
4. Train the same CNN architecture with on-the-fly augmentation.
5. Evaluate both models on the same untouched test set.
6. Display metrics and plots in a Streamlit interface.
7. Upload an image and obtain a predicted class and confidence score.

## Technologies
Python, Streamlit, TensorFlow/Keras, NumPy, Matplotlib, scikit-learn and Pillow.

## Results
Enter the actual test accuracy, test loss and training time shown by the app. Do not invent or assume that augmentation improves accuracy.

## Conclusion
Write the conclusion based on the measured test results. Discuss whether augmentation improved accuracy, reduced overfitting or had little effect.

## Limitations
CIFAR-10 contains small 32×32 images. The app uses conventional data augmentation rather than a GAN or diffusion model. Predictions on unrelated real-world photographs may be unreliable.
