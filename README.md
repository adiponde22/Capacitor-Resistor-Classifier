# Capacitor-Resistor-Classifier
This repository contains the code for training a machine learning model to classify electronic components, specifically resistors and capacitors, using image data. The "Resistor vs. Capacitor Classifier" utilizes the TensorFlow framework and implements a deep learning architecture based on the MobileNetV2 convolutional neural network.


# Key Features
Data Preprocessing: The code includes an ImageDataGenerator object that performs data preprocessing tasks such as rescaling the image pixel values. This ensures that the input data is properly prepared for training the model.

Model Architecture: The code defines a sequential model that consists of a pre-trained MobileNetV2 base model followed by global average pooling, fully connected layers, dropout regularization, and a softmax activation layer for multi-class classification.

Training and Validation: The code uses the flow_from_directory method to generate training and validation data on-the-fly from the provided directory structure. It splits the data into training and validation subsets, allowing the model to learn from the training data and evaluate its performance on the validation data.

Transfer Learning: The code loads the pre-trained MobileNetV2 model with weights from the ImageNet dataset. By setting trainable = False, the base model's weights are frozen, allowing the model to leverage the pre-trained features while only training the newly added layers.

Model Evaluation: During training, the model's performance is evaluated using the categorical cross-entropy loss function and accuracy metrics. The training and validation loss and accuracy are displayed and updated for each epoch, providing insights into the model's learning progress.

Model Saving: After training, the code saves the trained model as an HDF5 file (sensmodel.h5), allowing it to be easily loaded and used for inference or further fine-tuning in the future.

# Usage
To use the "Resistor vs. Capacitor Classifier" code:

Organize your resistor and capacitor images into separate directories (resistors and capacitors) within the pics directory. You can also include a none directory for images not belonging to either class.

Modify the IMG_HEIGHT, IMG_WIDTH, and BATCH_SIZE constants according to your requirements.

Run the code using a Python environment with TensorFlow and its dependencies installed.

The code will automatically generate training and validation data on-the-fly using the provided directory structure and preprocess the images using the ImageDataGenerator.

The model will be trained for the specified number of epochs, and the training and validation loss/accuracy will be displayed during training.

After training, the trained model will be saved as an HDF5 file (resistor_capacitor_none.h5) in the current directory.

Feel free to modify the code as needed, such as adjusting the model architecture, hyperparameters, or training settings, to suit your specific requirements.

# Contribution Guidelines
As this repository primarily contains code, contributions are welcome in the form of bug fixes, improvements, or additional functionality that enhances the Resistor vs. Capacitor Classifier. To contribute, please follow the standard guidelines:

Fork the repository and create a new branch for your contributions.
Ensure your code adheres to the repository's coding standards and best practices.
Document any modifications or additions thoroughly and include test cases where appropriate.
Submit a pull request detailing the purpose and changes made in your contribution.
By contributing to this repository, you contribute to the development of an accurate and efficient classifier, benefiting electronics enthusiasts, professionals, and researchers.
