# Applied Machine Learning Systems (ELEC0134)

**Student Number:** SN22145579

This repository contains the implementation for the ELEC0134 Applied Machine Learning Systems coursework.  
The project benchmarks classical machine learning and deep learning models on the BreastMNIST dataset.

---

## Project Structure
```
AMLS_25_26_SN22145579/
│
├── A/ # Model A – Classical ML (SVM)
│ ├── data_acquisition_a.py
│ └── train_val_svm.py
│
├── B/ # Model B – Deep Learning (CNN)
│ ├── data_acquisition.py
│ ├── training.py
│ └── plotting.py
│
├── main.py # Main entry point
├── Datasets/ 
└── README.md

```
---

## Requirements

The following Python packages are required:

- numpy < 2
- scikit-learn
- torch
- torchvision
- medmnist
- matplotlib

**Note:** NumPy is constrained to `<2` to ensure binary compatibility with PyTorch and MedMNIST.

Dependencies can be installed using:


pip install -r requirements.txt

How to Run

From the root directory of the project, run:
python main.py
