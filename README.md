# Gender-Detection
This project involves predicting gender from grayscale images using a Convolutional Neural Network (CNN). The dataset is preprocessed, and a deep learning model is trained to classify gender (binary classification: Male/Female).

# Overview

This project builds a deep learning pipeline to classify gender from facial images using TensorFlow and Keras. The model is trained on a dataset of 48x48 grayscale images. The training and testing process includes steps such as data normalization, data augmentation, and evaluation of model performance.

The following Python libraries are required:

numpy
pandas
opencv-python
matplotlib
tensorflow
scikit-learn

# Data Preprocessing

The dataset is loaded and preprocessed using the load_dataset function:
Image data is extracted, normalized, and reshaped to include a channel dimension.
Labels (gender) are converted to categorical format for binary classification.

# Training the Model

The CNN model is built using Keras and consists of:
Two convolutional layers with ReLU activation and max-pooling.
Dropout layers to prevent overfitting.
A dense layer with softmax activation for gender classification.

# Evaluating the Model

The model is evaluated on the test set using:
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
