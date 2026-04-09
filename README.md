#  Vehicle Damage Classification using Deep Learning

##  Overview

This project focuses on **automated vehicle damage classification** using deep learning. The model analyzes uploaded images of cars and predicts the type and location of damage, enabling applications in **insurance automation, inspection systems, and fleet management**.

The system is built as an **end-to-end ML application**, with DL methodological integrations of:

* Deep Learning model (ResNet50)
* FastAPI backend
* Streamlit frontend

---

##  Problem Statement

Manual vehicle damage inspection is:

* Time-consuming
* Subjective
* Prone to human error

This project aims to:

> Automate damage detection and classification using computer vision.

---

##  Approach & Methodology

### 1. Data Understanding

The dataset consists of vehicle images categorized into:

* Front Breakage
* Front Crushed
* Front Normal
* Rear Breakage
* Rear Crushed
* Rear Normal

This makes it a **multi-class image classification problem (6 classes)**.

---

### 2. Data Preprocessing

####  Image Transformations

* Resize to **224×224** (standard input for ResNet)
* Normalize using **ImageNet statistics**

```python
transforms.Normalize(
  mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.225]
)
```

#### 💡 Why?

* Ensures compatibility with pretrained models
* Improves convergence and stability

---

### 3. Model Selection

#### 🔹 ResNet50 (Transfer Learning)

We used a pretrained **ResNet50** model and modified it:

* Froze initial layers → retain learned features
* Fine-tuned deeper layers → adapt to damage patterns
* Replaced final fully connected layer → match 6 classes

```python
self.model = models.resnet50(weights='DEFAULT')
```

#### 💡 Why ResNet50?

* Proven performance in image tasks
* Deep architecture with residual connections
* Prevents vanishing gradient problem

---

### 4. Training Strategy

#### 🔁 Experiments Performed

* Baseline model (no tuning)
* Fine-tuning last layers
* Regularization using Dropout
* Hyperparameter tuning

#### ⚙️ Optimization

* Loss: CrossEntropyLoss
* Optimizer: Adam / SGD (depending on experiment)

#### 💡 Why Transfer Learning?

* Reduces training time
* Works well with limited data
* Leverages pretrained knowledge

---

### 5. Hyperparameter Tuning

Performed tuning to improve model performance:

* Learning rate
* Batch size
* Dropout rate

#### 💡 Why?

To balance:

* Underfitting vs Overfitting
* Speed vs Accuracy

---

### 6. Model Output

The model predicts one of the following:

```python
[
 'Front Breakage',
 'Front Crushed',
 'Front Normal',
 'Rear Breakage',
 'Rear Crushed',
 'Rear Normal'
]
```

---

## 🏗️ System Architecture

```
User Upload (Streamlit UI)
        ↓
Image Preprocessing
        ↓
Model Inference (ResNet50)
        ↓
Prediction Output
        ↓
Displayed on UI / API Response
```

---

##  Tech Stack

| Component        | Technology               |
|------------------| ------------------------ |
| DL Framework     | PyTorch                  |
| Backend          | FastAPI                  |
| Frontend         | Streamlit                |
| Image Handling   | PIL                      |
| Deployment Ready | Render / Streamlit Cloud |

---

##  Project Structure

```
project/
│
├── app.py               # Streamlit UI
├── server.py            # FastAPI backend
├── model_helper.py      # Model loading & prediction logic
├── saved_model.pth      # Trained model
├── requirements.txt
└── README.md
```

---

##  How to Run Locally

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App

```bash
streamlit run app.py
```

### 4. Run FastAPI Server (optional)

```bash
uvicorn server:app --reload
```

---

##  API Endpoint

### POST `/predict`

**Input:** Image file
**Output:**

```json
{
  "prediction": "Front Crushed"
}
```

---





##  Learnings

Through this project:

* Applied transfer learning in real-world scenario
* Built end-to-end ML system
* Integrated model with frontend & backend
* Understood deployment challenges

---

##  Conclusion

This project demonstrates how deep learning can be used to automate **vehicle damage classification**, reducing manual effort and improving efficiency in inspection systems.

---

