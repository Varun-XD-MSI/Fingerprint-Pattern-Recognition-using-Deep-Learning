# 🔍 Fingerprint Pattern Recognition using Deep Learning

This project implements a deep learning-based system to classify fingerprint images into three primary pattern types:

* 🟢 **Arch**
* 🔵 **Loop**
* 🟣 **Whorl**

The model is built using transfer learning with a pretrained **ResNet18** architecture and deployed through a simple web interface using Streamlit.

---

## 🚀 Features

* Image classification using deep learning (CNN)
* Transfer learning with pretrained ResNet18
* Data augmentation for improved generalization
* Weighted loss to handle class confusion
* Model evaluation using validation and test sets
* Confusion matrix for performance analysis
* Interactive web app for real-time predictions

---

## 🧠 Tech Stack

* Python
* PyTorch
* Torchvision
* Streamlit
* NumPy, Matplotlib, Scikit-learn

---

## 📂 Dataset

* Source: NIST DB4 Fingerprint Dataset
* Classes:

  * Arch
  * Loop
  * Whorl

Dataset is organized into:

* `train_set/`
* `val_set/`
* `test_set/`

---

## ⚙️ Model Details

* Architecture: ResNet18 (pretrained)
* Input size: 224 × 224
* Optimization: Adam optimizer
* Loss Function: Weighted CrossEntropyLoss
* Training Strategy: Transfer learning (frozen backbone)

---

## 📊 Results

| Metric              | Value |
| ------------------- | ----- |
| Validation Accuracy | ~81%  |
| Test Accuracy       | ~78%  |

---

## 🔍 Key Insights

* The model performs well overall but shows confusion between **loop and whorl patterns** due to their visual similarity.
* A confusion matrix was used to analyze misclassifications and understand model limitations.
* Fingerprint classification is inherently challenging due to partial prints and overlapping ridge structures.

---

## 📉 Confusion Matrix

<img width="577" height="433" alt="Confusion" src="https://github.com/user-attachments/assets/05fe63a8-49ac-4669-a459-4cdb5dd6226c" />


---

## 💻 Web App

<img width="982" height="676" alt="App" src="https://github.com/user-attachments/assets/9faf51ad-e102-416b-9bcf-88763a128df6" />


A Streamlit-based interface allows users to:

* Upload a fingerprint image
* Get predicted pattern instantly
* View prediction confidence

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-usernam/fingerprint-classifier.git](https://github.com/Varun-XD-MSI/Fingerprint-Pattern-Recognition-using-Deep-Learning)
cd fingerprint-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📌 Future Improvements

* Improve accuracy with larger and more diverse datasets
* Implement ridge enhancement preprocessing
* Use advanced architectures (EfficientNet, Vision Transformers)
* Deploy permanently on cloud platforms

---

## 🙌 Acknowledgements

* NIST Fingerprint Dataset
* PyTorch and Torchvision libraries

---

## 📬 Contact

Feel free to connect or reach out for feedback and collaboration!
