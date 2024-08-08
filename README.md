COVID-19 Radiography Classification
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify chest X-ray images as either COVID-19 positive or Normal. The project also includes a web interface built with Gradio, allowing users to upload images and receive predictions along with confidence scores.

Table of Contents
Project Structure
Requirements
Installation
Usage
Model Training
Gradio Interface
Results
Acknowledgments
Project Structure
plaintext
Copy code
COVID-19_Radiography_Classification/
├── filename.py   # Main script for training and running the model
├── README.md                   # Project documentation
├── training_history.png        # Saved plot of training history (accuracy and loss)
└── requirements.txt            # List of dependencies
Requirements
Ensure you have the following dependencies installed:

Python 3.7+
TensorFlow 2.x
OpenCV 4.x
Matplotlib
Pandas
Numpy
Scikit-learn
Gradio
Installation
Clone this repository to your local machine.

Navigate to the project directory.

Install the required dependencies using the following command:

bash
Copy code
pip install -r requirements.txt
Usage
Running the Script:

To train the model and launch the Gradio interface, run the following command:

bash
Copy code
python filename.py
Uploading Images for Classification:

Open the Gradio interface in your browser.
Upload one or more chest X-ray images.
Click the "Classify" button to get predictions and view the results.
Viewing Training History:

Navigate to the "Training History" tab in the Gradio interface to view the accuracy and loss plots from the model training.
Model Training
The CNN model is trained on a dataset of chest X-ray images from the COVID-19 Radiography Dataset.

Key Model Features:
Input Shape: 50x50x3 (RGB images resized to 50x50 pixels)
Layers:
Convolutional layers with ReLU activation
Max Pooling layers
Dense layers with ReLU activation
Output layer with Sigmoid activation for binary classification
Training Process:
Optimizer: Adam
Loss Function: Binary Crossentropy
Metrics: Accuracy
Epochs: 5
Gradio Interface
The Gradio interface allows users to interact with the model easily. It provides the following features:

Image Classification:

Upload multiple chest X-ray images for classification.
View predictions along with confidence scores.
Display images with overlaid predictions.
Training History:

View training accuracy and loss plots.
Results
The model provides predictions on whether a chest X-ray image is COVID-19 positive or Normal. The results are displayed in a tabular format along with confidence scores, and the predictions are overlaid on the images for easy interpretation.

Acknowledgments
This project uses the COVID-19 Radiography Dataset for training and testing.
The CNN model is implemented using TensorFlow and Keras, with a user interface provided by Gradio.
