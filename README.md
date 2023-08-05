AutoEns: An API-Based Intelligent Framework for Automated Ensemble Learning Model Development and Deployment
============================================================================================================

Description
-----------

AutoEns is an intelligent API-based framework designed for automated ensemble learning model development and deployment. It streamlines the process of creating, training, and using ensemble models for various machine learning tasks.

Getting Started
-------------
### 1. Create a Virtual Environment
A virtual environment helps isolate your project's dependencies from the system-wide Python installation.
#### On Linux/macOS
    python3 -m virtualenv venv
    source venv/bin/activate

#### On Windows
    python -m virtualenv venv
    virtualenv\Scripts\activate

### 2. Install Dependencies
Once the virtual environment is activated, use pip to install the required packages from the requirements.txt file.

    pip install -r requirements.txt

API Endpoints
-------------

### 1. Analyze Dataset

Use this endpoint to analyze the dataset and prepare it for modeling.

`curl --location 'http://127.0.0.1:5000/analyze' --form 'dataset="HeartFailure"' --form 'label="HeartDisease"' `

### 2. Model Development

Use this endpoint to develop an ensemble learning model using the pre-processed dataset.

`curl --location 'http://127.0.0.1:5000/modeling' --form 'dataset="HeartFailure"' --form 'label="HeartDisease"' `

### 3. Prediction

Use this endpoint to make predictions using the trained ensemble model.

`curl --location 'http://127.0.0.1:5000/predict' --form 'model="HeartFailure"' --form 'data="46,M,ASY,120,277,0,Normal,125,Y,1,Flat"' --form 'columns="Age,Sex,ChestPainType,RestingBloodPressure,Cholesterol,FastingBloodSugar,RestingElectrocardiography,MaxHeartRate,ExerciseAngina,Oldpeak,STSlope"' `

Dataset Citation
-------
As an example, this project uses the "Heart Failure Prediction Dataset" provided by fedesoriano in September 2021. Retrieved in July 2023 from Kaggle: https://www.kaggle.com/fedesoriano/heart-failure-prediction.

Contributing
-------
We welcome contributions from the community. If you find any issues or have suggestions to improve AutoEns, please feel free to open an issue or submit a pull request. Let's collaborate to make ensemble learning more accessible and effective for everyone.

License
-------
This project is licensed under the GNU General Public License version 3 (GPLv3) - see the LICENSE file for details.