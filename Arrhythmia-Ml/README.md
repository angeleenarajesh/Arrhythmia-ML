# Arrhythmia Detection using Machine Learning (SVM + PCA)

## About this Project

This project focuses on detecting **cardiac arrhythmia (abnormal heartbeats)** using machine learning techniques applied to ECG signal data.

Using the **INCART 2-lead Arrhythmia Dataset**, I built a classification model to distinguish between **normal (N)** and **abnormal heart activity**, aiming to explore how effectively machine learning can assist in medical signal analysis.

---

##  What I Did

* Processed raw ECG dataset and extracted features
* Converted multi-class labels into a **binary classification problem**

  * **0 → Normal**
  * **1 → Abnormal**
* Handled missing values using **mean imputation**
* Applied **feature scaling (StandardScaler)**
* Reduced dimensionality using **PCA (95% variance retained)**
* Trained an **SVM (Support Vector Machine)** with RBF kernel
* Used **GridSearchCV** to optimize hyperparameters
* Evaluated performance using multiple metrics

---

## Model Details

* **Model:** Support Vector Machine (RBF Kernel)
* **Dimensionality Reduction:** PCA
* **Best Parameters:**

  * C = 10
  * gamma = scale

---

## Results

* **Accuracy:** 99.65%
* **ROC-AUC Score:** 0.998

###  Classification Report

* High precision and recall for both classes
* Strong ability to detect abnormal heartbeats
* Balanced performance across dataset

This indicates that the model is highly effective in distinguishing between normal and abnormal ECG patterns.

---

## Visualizations

### 🔹 Confusion Matrix

Shows how well the model distinguishes between classes.

![Confusion Matrix](results/confusion%20matrix.png)

---

### 🔹 Class Distribution

Displays the balance between normal and abnormal samples.

![Class Distribution](results/class%20distribution.png)

---

## Key Insights

* PCA reduced features from **32 → 14**, improving efficiency without losing important information
* SVM performed extremely well for this classification problem
* Proper preprocessing (scaling + imputation) significantly improved performance
* The dataset is well-structured, allowing high classification accuracy

---

## Important Note

Although the model achieves very high accuracy, this does **not directly translate to real-world medical deployment**. Clinical validation and domain-specific evaluation are essential for practical applications.

---

## Future Improvements

* Use deep learning models (CNN/LSTM) for ECG signal analysis
* Work with raw signal data instead of extracted features
* Perform cross-dataset validation
* Explore multi-class classification (different arrhythmia types)

---

## Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

---

## About Me

MSc Physics student exploring the intersection of **data science, signal processing, and real-world applications**.

Interested in applying computational techniques to both **astrophysics and biomedical data analysis**.

---

## Final Note

This project highlights how machine learning can be applied beyond traditional domains, showing its potential in **healthcare and signal-based analysis**.

It reflects my interest in using data-driven approaches to solve meaningful, real-world problems.
