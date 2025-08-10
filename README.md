# Hand-Written Digit Classification

## Project Overview

This is a convolutional neural network (CNN) trained on the MNIST (10 classes, 60000 samples) dataset, to be able to classify hand-written digits.

## Training

The model is a CNN with 2 convolutional layers and 2 fully connected layers.
- Convolutional Layer 1 (followed by ReLU activation and Max Pooling)
- Convolutional Layer 2 (followed by ReLU activation and Max Pooling)
- Flattening
- Fully Connected Layer 1 (with ReLU activation)
- Fully Connected Layer 2 (with Softmax activation)

Metrics:
    - Lowest loss on test dataset: 0.0340
    - Highest accuracy on test dataset: 99.20%

## Inference

This project also has a basic interface for model evaluation at [src/main.py](src/main.py)

## File Structure

```
├───data
│   └───MNIST
│       └───raw
├───models
├───notebooks
└───src
```