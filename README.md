# AMLS_24_25_SN22145579
Student Number: SN22145579

This repository contains the implementation for the ELEC0134 Applied Machine Learning Systems coursework, benchmarking classical machine learning and deep learning models on the BreastMNIST dataset.

Project Structure:
AMLS_25_26_SN22145579
|
|
|__A\
|   |
|   |__data_acquisition_a.py
|   |
|   |__train_val_svm.py
|
|
|__B\
|   |
|   |__ data_acquisition.py
|   |
|   |__training.py
|   |__ plotting.py
|
|__ main.py
|
|
|__Datasets\
|
|
|__README.md

The following packages are required
numpy<2
scikit-learn
torch
torchvision
medmnist
matplotlib

NumPy is constrained to <2 to ensure compatability.

They can be installed using the requirement.txt file 
pip install -r requirement.txt

Code can be run from root directory on the project using:
python main.py