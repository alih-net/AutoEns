AutoEns API Documentation
=========================

Welcome to the AutoEns API documentation. AutoEns is an intelligent framework designed for automated ensemble learning model development and deployment. This documentation provides an overview of the API endpoints, usage examples, and important instructions for getting started.

Table of Contents
-----------------
1. Introduction
2. Getting Started
3. API Endpoints
    - Analyze Dataset
    - Model Development
    - Prediction
4. Unit Tests
5. Dataset Citation
6. Contributing
7. License

Introduction
--------------
AutoEns is an API-based framework that automates the process of creating, training, and using ensemble learning models for various machine learning tasks. This documentation outlines the steps to set up and use the AutoEns framework.

Getting Started
----------------
To start using AutoEns, follow these steps:

1. Clone the Repository and Change Directory
Clone the AutoEns repository and navigate into it using the following commands:

git clone https://github.com/alih-net/AutoEns.git
cd AutoEns

2. Create a Virtual Environment
Set up a virtual environment to isolate dependencies:

On Linux/macOS:
python3 -m virtualenv venv
source venv/bin/activate

On Windows:
python -m virtualenv venv
venv\Scripts\activate

3. Install Dependencies
Install the required packages from the requirements.txt file:

pip install -r requirements.txt

API Endpoints
---------------
1. Analyze Dataset
Analyze and prepare the dataset for modeling.

Request:
- Endpoint: /analyze
- Method: POST
- Parameters:
    - dataset: Name of the dataset (string)
    - label: Name of the target variable (string)

Example:
curl --location 'http://127.0.0.1:5000/analyze' --form 'dataset="HeartFailure"' --form 'label="HeartDisease"'

2. Model Development
Develop an ensemble learning model using the pre-processed dataset.

Request:
- Endpoint: /modeling
- Method: POST
- Parameters:
    - dataset: Name of the dataset (string)
    - label: Name of the target variable (string)

Example:
curl --location 'http://127.0.0.1:5000/modeling' --form 'dataset="HeartFailure"' --form 'label="HeartDisease"'

3. Prediction
Make predictions using the trained ensemble model.

Request:
- Endpoint: /predict
- Method: POST
- Parameters:
    - model: Name of the trained model (string)
    - data: Comma-separated input data values (string)
    - columns: Comma-separated column names corresponding to the input data (string)

Example:
curl --location 'http://127.0.0.1:5000/predict' --form 'model="HeartFailure"' --form 'data="46,M,ASY,120,277,0,Normal,125,Y,1,Flat"' --form 'columns="Age,Sex,ChestPainType,RestingBloodPressure,Cholesterol,FastingBloodSugar,RestingElectrocardiography,MaxHeartRate,ExerciseAngina,Oldpeak,STSlope"'

Unit Tests
-----------
To run unit tests, execute the following command:

python -m unittest discover -s tests -p "*_test.py"

Dataset Citation
----------------
The "Heart Failure Prediction Dataset" used in this project is provided by fedesoriano (September 2021) and retrieved from Kaggle: Heart Failure Prediction Dataset

Contributing
------------
We welcome contributions to AutoEns. If you encounter issues or have suggestions for improvement, feel free to open an issue or submit a pull request.

License
-------
This project is licensed under the GNU General Public License version 3 (GPLv3). For more details, refer to the LICENSE file.
