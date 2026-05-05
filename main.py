from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()


# Load model
model = joblib.load("./models/rf_model_classification_delay_risk.pkl")
reg_model = joblib.load("./models/rf_model_regression_wait_time.pkl")
features = joblib.load("./models/feature_list.pkl")
threshold = joblib.load("./models/classification_threshold.pkl")

model_encoder = joblib.load("./models/le_model.pkl")
issue_encoder = joblib.load("./models/le_issue_category.pkl")
service_encoder = joblib.load("./models/le_service_type.pkl")


class Job(BaseModel):
    date: str
    appointment_time: str
    # arrival_time: str
    # work_start_time: str
    # work_finish_time: str

    model: str
    issue_category: str
    service_type: str

    issue_complexity: float
    issue_duration_est: float
    vehicle_age_months: float
    mileage: float

    num_technicians: float
    service_slots: float
    backlog_size: float
    appointments_per_day: float
    demand_capacity_ratio: float


class RequestBody(BaseModel):
    job: Job


# Feature Engineering

def time_to_minutes(t):
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def build_features(job: Job):
    appt = time_to_minutes(job.appointment_time)
    # arrive = time_to_minutes(job.arrival_time)
    # start = time_to_minutes(job.work_start_time)
    # finish = time_to_minutes(job.work_finish_time)

    # arrival_delay = arrive - appt
    # queue_wait = max(start - arrive, 0)
    # actual_duration = max(finish - start, 0)
    # tech_behind = max(start - appt, 0)

    df = pd.DataFrame([{
        # encoded categorical
        "model_enc": model_encoder.transform([job.model])[0],
        "issue_category_enc": issue_encoder.transform([job.issue_category])[0],
        "service_type_enc": service_encoder.transform([job.service_type])[0],

        # core numeric inputs
        "issue_complexity": job.issue_complexity,
        "issue_duration_est": job.issue_duration_est,
        "vehicle_age_months": job.vehicle_age_months,
        "mileage": job.mileage,

        "num_technicians": job.num_technicians,
        "service_slots": job.service_slots,
        "backlog_size": job.backlog_size,
        "appointments_per_day": job.appointments_per_day,
        "demand_capacity_ratio": job.demand_capacity_ratio,

        # engineered features (must match training)
        # "arrival_delay_mins": arrival_delay,
        # "queue_wait_mins": queue_wait,
        # "tech_behind_at_start": tech_behind,
        "appt_mins": appt,

        # simple engineered flags
        # "customer_late": int(arrival_delay > 10),
        # "customer_very_late": int(arrival_delay > 30),
        # "tech_behind_flag": int(tech_behind > 30),

        # safe defaults for missing engineered features
        "day_running_behind_mins": 0,
        "rolling_tech_overrun_avg": 0,
        "appointment_hour": appt // 60,
        "appointment_slot_mins": appt % 60,
    }])

    # ensure all model features exist
    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]



# Prediction endpoint

@app.post("/predict")
def predict(req: RequestBody):
    job = req.job

    X = build_features(job)

    # classification
    prob = model.predict_proba(X)[0][1]
    delay_risk = int(prob >= threshold)

    # regression
    wait_time = float(reg_model.predict(X)[0])

    return {
        "delay_risk_probability": float(prob),
        "delay_risk": delay_risk,
        "predicted_wait_time": wait_time
    }

# PREDICT ENDPOINT
@app.post("/predict")
def predict(req: RequestBody):
    try:
        job = req.job
        X = build_features(job)

        prob = model.predict_proba(X)[0][1]
        delay_risk = int(prob >= threshold)
        wait_time = float(reg_model.predict(X)[0])

        return {
            "delay_risk_probability": float(prob),
            "delay_risk": delay_risk,
            "predicted_wait_time": wait_time
        }

    except Exception as e:
        return {
            "error": str(e)
        }
