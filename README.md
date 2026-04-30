# 🧠 Multi-Architecture Deep Learning Pipeline


<img width="1701" height="604" alt="image" src="https://github.com/user-attachments/assets/61d14c28-ef39-4a2b-aa95-9dc289265385" />


This repository contains a comprehensive machine learning research project. I developed a universal Python-based pipeline capable of training and evaluating neural networks across various data domains: computer vision (images), time-series forecasting, and tabular data classification/regression. 

Supported Architectures
Depending on the specific task and dataset, the pipeline utilizes different neural network architectures:
* **CNN (Convolutional Neural Networks):** Configured for spatial feature extraction in image processing tasks.
* **RNN & LSTM (Long Short-Term Memory):** Implemented for sequential data and time-series analysis to solve the vanishing gradient problem.
* **MLP (Multi-Layer Perceptron):** Used for classical tabular data processing.

Core Features
* **Data Preprocessing Module:** Automated value normalization, dataset splitting (train/test/validation), and data formatting.
* **Hyperparameter Tuning:** Configurable learning rates, epochs, and loss functions for optimal model convergence.
* **Performance Evaluation:** Built-in metrics calculation to track model accuracy and loss during the training lifecycle.

Tech Stack
* **Language:** Python
* **Deep Learning Frameworks:** PyTorch, TensorFlow
* **Data Processing & ML:** Pandas, NumPy, scikit-learn
* **Other:** requests, SciPy
