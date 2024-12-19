# HAR-and-Object-Detection-Analysis
This repository contains an in-depth analysis and implementation of Human Activity Recognition (HAR) using the UCI dataset and Object Detection using the YOLO model.


# Human Activity Recognition (HAR) and Object Detection Project

This repository provides an in-depth implementation of Human Activity Recognition (HAR) using the UCI dataset and Object Detection using the YOLO model. The project consists of three primary tasks, exploring both traditional machine learning and advanced deep learning techniques.

---

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Task 1: Classic Machine Learning Models on Featurized Data](#task-1-classic-machine-learning-models-on-featurized-data)
- [Task 2: Deep Learning Models on Raw Sensor Data](#task-2-deep-learning-models-on-raw-sensor-data)
- [Task 3: Object Detection using YOLO Model](#task-3-object-detection-using-yolo-model)
- [Technologies and Tools](#technologies-and-tools)
- [Deliverables](#deliverables)

---

## Introduction
The Human Activity Recognition (HAR) dataset from the University of California, Irvine (UCI) comprises two types of data:
- **Featurized Data**: Pre-extracted features representing activities.
- **Raw Inertial Sensor Data**: Accelerometer and gyroscope readings from wearable devices.

This project demonstrates the application of classic machine learning and deep learning models on HAR data, along with an object detection task using the YOLO model.

---

## Project Structure
- `Task1_Classic_ML_Models.ipynb`: Implementation of machine learning models on featurized data.
- `Task2_Deep_Learning_Models.ipynb`: Deep learning models trained on raw sensor data.
- `Task3_Object_Detection_YOLO.ipynb`: Custom object detection pipeline using YOLO.
- `datasets/`: Contains the HAR and object detection datasets.
- `annotations/`: Annotation files for YOLO training.
- `results/`: Performance metrics and visualizations.

---

## Task 1: Classic Machine Learning Models on Featurized Data
- **Objective**: Classify human activities using the pre-extracted features from the HAR dataset.
- **Models Used**: Random Forest, Decision Tree, Logistic Regression, AdaBoost.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score.

**Steps**:
1. Data exploration and preprocessing.
2. Model training and hyperparameter tuning.
3. Comparison of model performances.

---

## Task 2: Deep Learning Models on Raw Sensor Data
- **Objective**: Classify human activities using raw accelerometer and gyroscope data.
- **Models Used**: 1D Convolutional Neural Networks (1D-CNN), Multi-Layer Perceptron (MLP), Long Short-Term Memory Network (LSTM).
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score.

**Steps**:
1. Preprocessing raw sensor data.
2. Designing and training deep learning models.
3. Performance evaluation and comparison.

---

## Task 3: Object Detection using YOLO Model
- **Objective**: Detect objects using a custom YOLO model.
- **Steps**:
  1. Capturing and annotating images of objects.
  2. Training the YOLO model on the custom dataset.
  3. Evaluating the YOLO model using a confusion matrix.

---

## Technologies and Tools
- **Programming Language**: Python
- **Machine Learning Libraries**: Scikit-learn, XGBoost
- **Deep Learning Frameworks**: TensorFlow, Keras, PyTorch
- **Object Detection Tools**: YOLOv5
- **Visualization Tools**: Matplotlib, Seaborn
- **Dataset Management**: Pandas, NumPy

---

## Deliverables
1. Jupyter notebooks for each task:
   - `Task1_Classic_ML_Models.ipynb`
   - `Task2_Deep_Learning_Models.ipynb`
   - `Task3_Object_Detection_YOLO.ipynb`
2. Documentation with step-by-step explanations.
3. Visualizations for insights and performance analysis.
4. Summarized findings in markdown cells.

---

## Summary
This project showcases the comparison between classic machine learning models and deep learning techniques for activity recognition and demonstrates object detection using a YOLO model. The repository provides a comprehensive exploration of the dataset, model implementations, and performance evaluations.

