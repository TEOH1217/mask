# Lab 2 Enhancement: Fashion MNIST with PyTorch

## Overview
This lab enhancement focuses on implementing Convolutional Neural Networks (CNN) and Artificial Neural Networks (ANN) for classifying images in the Fashion MNIST dataset using PyTorch. The primary objective is to explore and compare the classification performance of CNN and ANN models, assessing key metrics such as training and validation loss, as well as accuracy.

## Contributor

**Teoh Chee Hong**  
Student ID: 1221303824

## Main Contributions
Model Implementation: Convolutional Neural Network (CNN) and Artificial Neural Network (ANN) models optimized for the sleek MNIST dataset were developed and implemented.The CNN model utilizes a convolutional layer to extract hierarchical features from an image, while the ANN model utilizes a densely connected layer to learn patterns in the dataset.

Performance Comparison: The performance of the CNN and ANN models is systematically evaluated and compared using rigorous metrics such as training and validation losses and accuracy scores. This comparative analysis provides insight into how each model architecture handles the complexity of the fashion MNIST classification task.

Strengths and Weaknesses Analysis: The inherent strengths and weaknesses of CNNs and ANNs are examined in detail when applied to fashion MNIST classification. This analysis considers factors such as computational efficiency, interpretability of results, and resilience to overfitting, providing a nuanced understanding of each model's suitability for the task.

Visualization: Visual representations of performance metrics and model structure are generated for the CNN and ANN architectures. These visualizations enhance understanding of model behavior, facilitate communication of findings, and illustrate how architectural differences affect performance results in the context of fashioning MNIST classification tasks.

## Result
### Training the CNN Model
![image](https://github.com/Heng12312/Machine-Learning-Project/assets/66999030/3a5af800-d241-4ad5-adca-d8613940f0f8)

### Training the ANN Model
![image](https://github.com/Heng12312/Machine-Learning-Project/assets/66999030/b14be20c-7852-4454-931b-abe0a387048d)

Training and Validation Loss: This section graphically shows how the training and validation loss of Convolutional Neural Network (CNN) and Artificial Neural Network (ANN) models progress over time. The graphs visually depict how the models converge or diverge during training to minimize losses, providing insight into their respective learning capabilities and generalization potential.

Training and Validation Accuracy: The graphs presented in this section of the results depict how the training and validation accuracy metrics of the CNN and ANN architectures have evolved over time. These graphs visualize the extent to which each model learns to correctly classify fashion MNIST images during training, and the extent to which it can effectively generalize to unseen validation data. Accuracy trends help to understand how well the models performed in terms of classification accuracy during training.

![image](https://github.com/Heng12312/Machine-Learning-Project/assets/66999030/3ed83d32-11f1-40b2-9540-309f3f5f1542)
![image](https://github.com/Heng12312/Machine-Learning-Project/assets/66999030/bd58230e-47fe-4402-94db-b31ab7050914)

### Comparison with CNN and ANN
1. Training and Validation Loss

CNN: Both training and validation losses are steadily decreasing with little fluctuation. Validation loss increases slightly near the end, indicating possible overfitting.
ANN: The training loss decreases steadily, but the validation loss fluctuates more, indicating a less stable learning process.

2. Training and Validation Accuracy

CNN: Training accuracy improves rapidly. Validation accuracy is slightly lower, indicating that the model generalizes well, but may begin to overfit as the gap eventually widens.
ANN: Training Accuracy Continues to Improve Rapidly, but will Be slightly lower than CNN. Validation accuracy is slightly lower and fluctuates, indicating that the model is not as stable as the CNN.
