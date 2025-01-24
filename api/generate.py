import requests
import random
import time

url = "http://127.0.0.1:8000/predict/"

for _ in range(10):  # Generate 10 records
    data = {
        "PatientID": f"Patient-{random.randint(1, 100)}",
        "WBC": random.uniform(4, 20),
        "Glucose": random.uniform(70, 250),
        "Temp": random.uniform(36.5, 39),
        "HR": random.uniform(60, 150),
        "Resp": random.uniform(12, 25),
        "Creatinine": random.uniform(0.6, 1.9),
        "Bilirubin_total": random.uniform(0.1, 1.5),
        "BUN": random.uniform(7, 25),
        "FiO2": random.uniform(0.21, 0.4),
        "SBP": random.uniform(90, 180),
        "MAP": random.uniform(70, 100)
    }
    response = requests.post(url, json=data)
    print(response.json())

print("10 records sent successfully.")