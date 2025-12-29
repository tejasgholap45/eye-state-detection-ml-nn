# 😴 Drowsiness Detection & Eye State Classification  
## 🧠 Machine Learning & Neural Network Project

---

## 📌 Project Overview

This project demonstrates an **end-to-end implementation of Machine Learning, Neural Networks, and Computer Vision** to solve real-world problems related to **eye state detection and drowsiness monitoring**.

The project is divided into two major parts:

1️⃣ **Eye State Classification using EEG signals**  
2️⃣ **Real-time Drowsiness Detection using a webcam**

---

## 🎯 Objectives

- 👁️ Predict eye state (Open / Closed) using EEG data  
- 🌲 Apply Machine Learning and Neural Network models  
- 🧠 Implement ANN and CNN architectures  
- 📷 Build a real-time webcam-based drowsiness detection system  
- 🚀 Deploy an interactive application using Streamlit  

---

## 📊 Datasets Used

### 🧬 EEG Eye State Dataset
- **Features:** 14 EEG signal values  
- **Target column:** `class`  
  - `0` → Eyes Open  
  - `1` → Eyes Closed  
- **Type:** Numerical tabular data  

### 🎥 Webcam Input
- Used for real-time drowsiness detection  
- Eye state inferred using eye landmark geometry (EAR method)

---

## 🔁 Methodology

### 🧩 Machine Learning Pipeline
```text
Data Loading
 → Data Exploration
 → Feature Scaling
 → Train-Test Split
 → Model Training
 → Model Evaluation

````

## 🤖 Models Implemented

### 🌲 1. Random Forest Classifier (Machine Learning)

* Handles noisy EEG data efficiently
* Used as a baseline ML model
* Evaluation Metrics:

  * ✅ Accuracy
  * 📊 Confusion Matrix
  * 📝 Classification Report

---

### 🧠 2. Artificial Neural Network (ANN)

* Fully connected feed-forward neural network
* Architecture:

```text
Input Layer (14 neurons)
 → Hidden Layer 1
 → Hidden Layer 2
 → Output Layer (1 neuron, Sigmoid)
```

* **Loss:** Binary Crossentropy
* **Optimizer:** Adam

---

### 🧠 3. Convolutional Neural Network (1D CNN)

* EEG signals treated as 1D sequences
* Automatically learns local feature patterns
* Architecture:

```text
Conv1D
 → MaxPooling
 → Dense
 → Output
```

---

### 😴 4. Real-Time Drowsiness Detection (Computer Vision)

* Webcam-based system (no EEG hardware required)
* Uses **MediaPipe Face Mesh**
* Calculates **Eye Aspect Ratio (EAR)**
* Detects prolonged eye closure and flags drowsiness
* Implemented as a **Streamlit live application**

---

## 🛠️ Technologies Used

* 🐍 Python
* 📊 Pandas, NumPy
* 🌲 Scikit-learn
* 🧠 TensorFlow / Keras
* 📷 OpenCV
* 🧩 MediaPipe
* 🌐 Streamlit

---

## 📂 Project Structure

```text
project/
│── app.py                  # Streamlit webcam application
│── eeg-eye-state.csv        # EEG dataset
│── cnn_eye_model.h5         # Trained CNN model
│── requirements.txt
│── README.md
│── LICENSE
```

---

## 📈 Key Results

* ✅ Random Forest achieved strong performance on EEG data
* ✅ ANN captured complex non-linear relationships
* ✅ CNN showed improved feature learning
* ✅ Webcam-based system successfully detected drowsiness in real time

---

## 🎓 Learning Outcomes

* Practical understanding of ML vs Deep Learning
* Hands-on experience with ANN and CNN
* Real-time computer vision system design
* Model evaluation and comparison
* Streamlit-based deployment of ML applications

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👤 Author
- **Name:** Tejas Gholap  
- **Domain:** Data Analytics | Machine Learning | Deep Learning  
- **Project Type:** Academic + Practical Implementation  

---

## 📝 Declaration

This project was developed as part of an academic **Machine Learning assignment** and demonstrates independent implementation of ML, Neural Network, and Computer Vision concepts.
