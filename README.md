# 🧠 Horse or Human Classifier

A simple Convolutional Neural Network (CNN) built using **TensorFlow and Keras** to classify images as either a **horse** or a **human**.

---

## 🚀 Project Overview
This project trains a CNN model on a dataset of horse and human images to automatically distinguish between the two classes.  
It includes:
- Model building and training
- Accuracy and loss visualization
- Single image prediction demo

---

## 🧩 Model Architecture
- **Conv2D + MaxPooling** layers for feature extraction  
- **Flatten + Dense** layers for classification  
- **Sigmoid** activation for binary output  

---

## 📂 Dataset
The dataset contains two folders:
horse-or-human/
┣ horses/
┗ humans/
horse-or-human-validation/
┣ horses/
┗ humans/

## Dataset
You can download the dataset from https://www.tensorflow.org/datasets/catalog/horse_or_human
and place it inside the project folder as:

datasets/
├── horse-or-human/
└── horse-or-human-validation/

Each folder should contain labeled images for training and validation.

---

## 🧪 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/horse-or-human-classifier.git
Install dependencies:

bash
Copy code
pip install tensorflow matplotlib numpy
Run the script:

bash

python horse_or_human_classifier.py
To test a single image, update the img_path at the bottom of the script and re-run.

📊 Training Results
The model achieves high accuracy on both training and validation sets after ~15 epochs.
Plots show the progression of accuracy and loss over time.

🧍🐴 Example Prediction
makefile
Copy code
Predicted: Human 🧍
🧰 Technologies Used
TensorFlow / Keras

NumPy

Matplotlib

✨ Author
Osama Al Attar
