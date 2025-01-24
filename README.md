# Sepsis Prediction Dashboard

## Overview
The Sepsis Prediction Dashboard is a web-based application designed to predict the risk of sepsis in patients based on clinical features. This project utilizes machine learning models for predicting the probabilities of infection, organ dysfunction, and sepsis, and stores these predictions in a database. A real-time dashboard visualizes the predictions and historical records.

## Features
- **Patient Data Input**: Allows users to input clinical data such as WBC, Glucose, Temperature, etc.
- **Sepsis Risk Prediction**: Predicts probabilities of infection, organ dysfunction, and overall sepsis risk using trained machine learning models.
- **Real-Time Visualization**: Displays a pie chart summarizing real-time sepsis predictions.
- **Historical Records**: View recent predictions stored in the database in a tabular format.

## Technologies Used
- **Backend**: FastAPI
- **Frontend**: HTML, CSS, JavaScript (Chart.js for visualization)
- **Database**: SQLite
- **Machine Learning Models**: Trained using `sklearn`, stored as `.pkl` files
- **Hosting**: Localhost during development

## Setup and Installation

### Prerequisites
- Python 3.8+
- `pip` (Python package manager)
- Recommended IDE: VS Code, PyCharm, or similar

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Navyatr13/Clinical_data_prediction.git
   cd sepsis-prediction-dashboard

2. Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
   
4. Prepare the database:

    ```bash
    python -c "import sqlite3; conn = sqlite3.connect('./data/sepsis_predictions.db'); conn.execute('VACUUM'); conn.close()"
5. Start the FastAPI server:

    ```bash
    python -m uvicorn api.app:app --reload
6. Access the application: Open a web browser and go to http://127.0.0.1:8000.

### Usage
1. Predict Sepsis Risk
Enter patient details such as WBC, Glucose, Temperature, etc., in the input form.
Click the "Predict" button.
View the prediction results and the patient's risk of sepsis.
2. View Real-Time Statistics
Monitor the Real-Time Sepsis Prediction Distribution pie chart to visualize the number of high-risk and low-risk sepsis predictions.
3. View Recent Records
Scroll down to view a table of recent predictions stored in the database, showing details of each patient's input and predictions.

## Key API Endpoints
### POST /predict/

- Predicts the probabilities of infection, organ dysfunction, and sepsis based on input data.
Request Body:
json
    ```{
      "PatientID": "Patient-1",
      "WBC": 8.5,
      "Glucose": 120,
      "Temp": 37.2,
      "HR": 85,
      "Resp": 18,
      "Creatinine": 1.0,
      "Bilirubin_total": 0.8,
      "BUN": 15,
      "FiO2": 0.21,
      "SBP": 110,
      "MAP": 85
    }```
Response:
    ```{
      "prob_infection": 0.25,
      "prob_organ_dysfunction": 0.45,
      "prob_sepsis": 0.3,
      "infection_flag": false,
      "organ_dysfunction_flag": false,
      "sepsis_flag": false
    }```
GET /records/

Fetches all prediction records stored in the database.
Response:
      ```json
      {
        "records": [
          [
            1, "Patient-1", 8.5, 120, 37.2, 85, 18, 1.0, 0.8, 15, 0.21, 110, 85, 0.25, 0.45, 0.3
          ],
          ...
        ]
      }```
GET /realtime-data/

Provides aggregated counts for high-risk and low-risk sepsis predictions.
Response:
    ```{
      "sepsis": 15,
      "non_sepsis": 25
    }```
## Testing
1. Generate Synthetic Data
Use the generate.py script to create synthetic patient records:

    ```bash
    python api/generate.py
2. Test Database
Run test_table.py to inspect the database contents:
    
    ```bash
    python scripts/test_table.py
3. Check API Responses
Use curl or Postman to test the /predict/, /records/, and /realtime-data/ endpoints.

### Future Improvements
- Enhance UI: Add more detailed visualizations and animations.
- Deploy to Cloud: Host the application on AWS, Azure, or GCP for public access.
- Add Authentication: Implement user roles for secure data handling.
- Model Retraining: Create an automated pipeline for retraining the ML models on new data.
- Dockerization: Containerize the application for easier deployment.
