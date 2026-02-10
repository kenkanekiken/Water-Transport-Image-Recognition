# Water-Transport-Image-Recognition
A deep learning image classification project that identifies different types of water transport vessels using a custom CNN and a ResNet50 pretrained model, with a Flask-based web interface for real-time predictions.

# CNN Water Transport Classification

## Overview
This project implements a deep learning–based image classification system to identify different types of water transport vessels. The model classifies images into five categories: **boat, cargo ship, cruise ship, hovercraft, and submarine**.

The project explores both a **custom Convolutional Neural Network (CNN)** and a **ResNet50 pretrained model using transfer learning**, and demonstrates how the trained model can be deployed through a **Flask web application** for real-time predictions.

---

## Objectives
- Build a CNN to understand fundamental image feature extraction
- Improve classification accuracy using **ResNet50 transfer learning**
- Apply data augmentation and fine-tuning to enhance generalisation
- Evaluate model performance on unseen test data
- Deploy the trained model using a Flask-based web interface

---

## Dataset
The dataset consists of images of water transport vessels grouped into five classes:
- Boat  
- Cargo Ship  
- Cruise Ship  
- Hovercraft  
- Submarine  

Offline data augmentation was applied to expand the training dataset by **5×**, helping to reduce overfitting and improve robustness.

---

## Model Architecture
### Custom CNN
- Multiple convolutional layers with ReLU activation
- MaxPooling layers for spatial downsampling
- Dropout to reduce overfitting
- Softmax output layer for multi-class classification

### ResNet50 (Transfer Learning)
- Pretrained on the ImageNet dataset
- Top classification layers removed (`include_top=False`)
- Used as a feature extractor
- Fine-tuned by unfreezing the last layers with a low learning rate

---

## Training and Evaluation
- Optimizer: Adam  
- Loss function: Categorical Cross-Entropy  
- Validation split: 80% training, 20% validation  
- Early stopping used to prevent overfitting  

**Final Performance:**
- Test Accuracy: ~92.9%
- Test Loss: ~0.49

These results indicate strong generalisation on unseen data.

---

## Web Application (Flask)
A Flask-based web application was developed to deploy the trained ResNet50 model.

### Features:
- Upload an image via a web browser
- Real-time classification
- Display of predicted class and confidence scores for all classes

This demonstrates how a deep learning model can be integrated into a real-world application.

├── dataset/ # Augmented training dataset
├── images/ # Original images and test images
├── models/ # Trained model files (.keras)
├── template/ # HTML templates
├── AIML_Project_*.ipynb # Training and evaluation notebooks
├── app.py # Flask web server
└── README.md

---

## Technologies Used
- Python
- TensorFlow / Keras
- ResNet50 (Transfer Learning)
- Flask
- NumPy

---

## Future Improvements
- Expand the dataset with more real-world images
- Experiment with other pretrained models such as EfficientNet
- Add confusion matrix and classification report visualisations
- Improve UI and add batch prediction support

---

## Author
**Kenneth**  
AI & Machine Learning Project
