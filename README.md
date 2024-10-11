# Satelite_Image_Classification



This repository provides a PyTorch-based solution for the Satelite Image Classification Challenge, which involves classifying natural scene images into multiple categories. The notebook implements a deep learning model using Convolutional Neural Networks (CNN) to solve the problem efficiently.

## Project Structure

```
├── CCN_image_pytorch-2.ipynb     # Main Jupyter notebook containing the model and implementation
├── README.md                     # This README file
├── data/                         # Folder to store the dataset (train/test)
├── models/                       # Folder to store the trained models
└── logs/                         # Folder to store training logs
```

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview

This notebook trains a CNN model using PyTorch for classifying natural scene images into the following six categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

The dataset is provided by the Intel Image Classification Challenge and consists of training, validation, and testing sets of images. The goal is to build a model that accurately predicts the category of unseen images.

## Installation

To run this notebook, you will need to install the following dependencies:

1. Python 3.x
2. PyTorch
3. torchvision
4. NumPy
5. Matplotlib
6. Pandas
7. Scikit-learn
8. tqdm (for progress bars)

To install the required libraries, run:

```bash
pip install torch torchvision numpy matplotlib pandas scikit-learn tqdm
```

## Dataset

The dataset can be downloaded from the [Kaggle Intel Image Classification Challenge](https://www.kaggle.com/puneet6060/intel-image-classification). Once downloaded, organize the dataset as follows:

```
data/
├── train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
├── test/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
└── validation/                   # (if applicable)
```

The dataset contains high-quality images that are classified into six categories. Ensure the dataset is downloaded and placed in the `data/` folder before running the notebook.

## Preprocessing

The notebook applies the following preprocessing steps:

- **Resize**: All images are resized to a fixed size (e.g., 150x150 pixels) using `torchvision.transforms.Resize`.
- **Normalization**: Pixel values are normalized using mean and standard deviation specific to the dataset.
- **Data Augmentation**: Techniques like random horizontal flips, random rotations, and random cropping are applied to the training set to increase the model's generalization ability.

The preprocessing pipeline is built using the `torchvision.transforms` library.

## Model Architecture

The model is a Convolutional Neural Network (CNN) built using PyTorch. The architecture includes:

- Multiple convolutional layers with batch normalization and ReLU activations.
- Max pooling layers for downsampling the feature maps.
- Dropout layers to reduce overfitting during training.
- Fully connected (dense) layers at the end for classification into six categories.

### Layers:

- **Convolutional Layer**: Extracts features from images using different filter sizes.
- **Batch Normalization**: Normalizes the output of the convolutional layers.
- **ReLU Activation**: Introduces non-linearity to the model.
- **Max Pooling**: Reduces the size of the feature maps.
- **Dropout**: Helps prevent overfitting by randomly deactivating some neurons during training.
- **Fully Connected Layers**: Used to map the extracted features to class probabilities.

## Training

The training process involves:

1. **Loss Function**: The model uses Cross-Entropy Loss for multi-class classification.
2. **Optimizer**: Adam optimizer is used with an initial learning rate, which is fine-tuned using a learning rate scheduler.
3. **Epochs**: The number of training epochs can be adjusted in the notebook.
4. **Batch Size**: The notebook uses a configurable batch size for training and testing the model.
5. **Validation**: The notebook includes a validation step at the end of each epoch to monitor the model's performance on unseen data.

### Checkpoints

Model checkpoints are saved after each epoch. This allows resuming training or performing further analysis on intermediate models.

## Evaluation

The model's performance on the test dataset is summarized by the confusion matrix shown below:

![Confusion Matrix](image.png)

### Confusion Matrix Overview

The confusion matrix shows the true labels (rows) against the predicted labels (columns). The darker the color, the higher the number of correct or incorrect classifications in that category. This matrix helps analyze the following:

- **True Positives (Diagonal Elements)**: These are the correctly classified examples for each category.
- **Misclassifications (Off-Diagonal Elements)**: These show how often an image from one category was classified into another category.

#### Key Insights from the Confusion Matrix:
1. **Forest** and **Mountain** categories have higher accuracy, with many correctly classified instances (336 for forest and 229 for mountain).
2. **Street** and **Sea** categories show some confusion with other classes, particularly misclassified as **Mountain** or **Glacier**.
3. The **Buildings** category had the highest misclassification rate, with a large number of images misclassified as **Sea** or **Mountain**.

### Classification Metrics:

- **Overall Accuracy**: 90% (replace with actual accuracy)
- **Precision, Recall, F1-Score**: These metrics for each category were computed (details can be extracted from the classification report).

### Misclassification Analysis:
- A significant number of images from the **Sea** class were misclassified as **Mountain** and vice versa, possibly due to visual similarities in their backgrounds (such as skies or open spaces).
- The **Street** category also shows confusion with **Mountain**, indicating that further refinement in the feature extraction process might be needed.

## Results

The final model achieves an accuracy of **X%** on the test dataset (replace with actual results). The following improvements can be made for better results:

- Tuning hyperparameters such as learning rate, batch size, and number of layers.
- Experimenting with pre-trained models (e.g., ResNet, VGG) for transfer learning.
- Implementing advanced data augmentation techniques.

## Future Work

Possible directions for improving the current model:

- **Transfer Learning**: Fine-tune a pre-trained model such as ResNet or VGG for this classification task.
- **Hyperparameter Tuning**: Explore different optimizers, learning rate schedules, and architectures to improve performance.
- **Advanced Data Augmentation**: Use techniques like CutMix or MixUp to improve generalization.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository, make your changes, and create a pull request. Contributions are welcome!

## Contact

For any questions or issues, please feel free to reach out at khushang20@gmail.com.
