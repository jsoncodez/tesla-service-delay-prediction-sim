# generate_models.py

import pandas as pd
import numpy as np
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                              f1_score, roc_auc_score,
                              mean_absolute_error, r2_score,
                              mean_absolute_percentage_error)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

os.makedirs('./models', exist_ok=True)
os.makedirs('./models/all', exist_ok=True)

print("="*65)
print("TESLA SERVICE DELAY — MODEL GENERATION")
print("="*65)

######  Load
print("\n[1/8] Loading dataset...")
df = pd.read_csv('./data/tesla_service_dataset.csv')
print(f"  Shape: {df.shape}")

######  Parse timestamps into numeric features
print("\n[2/8] Parsing timestamps...")

def hhmm_to_mins(series):
    """Convert HH:MM string column to minutes from midnight."""
    def parse(val):
        try:
            h, m = str(val).split(':')
            return int(h) * 60 + int(m)
        except:
            return np.nan
    return series.apply(parse)

df['appt_mins']   = hhmm_to_mins(df['appointment_time'])
df['arrive_mins'] = hhmm_to_mins(df['arrival_time'])
df['start_mins']  = hhmm_to_mins(df['work_start_time'])
df['finish_mins'] = hhmm_to_mins(df['work_finish_time'])

# Validate — fill any parse failures with medians
for col in ['appt_mins','arrive_mins','start_mins','finish_mins']:
    bad = df[col].isna().sum()
    if bad > 0:
        print(f"  Warning: {bad} unparseable values in {col}, filling with median")
        df[col] = df[col].fillna(df[col].median())

# Time-of-day buckets — useful categorical signal
def time_bucket(mins_series):
    """Label each slot: early_morning / mid_morning / lunch / afternoon / late"""
    def bucket(m):
        if   m < 9*60: return 0  # early morning  08:00–09:00
        elif m < 11*60: return 1  # mid morning     09:00–11:00
        elif m < 13*60: return 2  # pre/post lunch  11:00–13:00
        elif m < 15*60: return 3  # early afternoon 13:00–15:00
        else:
            return 4  # late afternoon  15:00+
    return mins_series.apply(bucket)

df['appt_time_bucket']   = time_bucket(df['appt_mins'])
df['start_time_bucket']  = time_bucket(df['start_mins'])

# Core derived time features
# These should already be in the CSV but recalculate to be safe
df['arrival_delay_mins'] = df['arrive_mins'] - df['appt_mins']
df['queue_wait_mins'] = (df['start_mins'] - df['arrive_mins']).clip(lower=0)
df['actual_duration_mins'] = (df['finish_mins'] - df['start_mins']).clip(lower=0)
df['total_time_in_shop_mins'] = (df['finish_mins'] - df['arrive_mins']).clip(lower=0)
df['tech_behind_at_start'] = (df['start_mins'] - df['appt_mins']).clip(lower=0)
df['duration_overrun_mins'] = df['actual_duration_mins'] - df['issue_duration_est']

# Was the customer late? Binary flags
df['customer_early'] = (df['arrival_delay_mins'] < -5).astype(int)
df['customer_on_time'] = ((df['arrival_delay_mins'] >= -5) & (df['arrival_delay_mins'] <= 10)).astype(int)
df['customer_late'] = (df['arrival_delay_mins'] > 10).astype(int)
df['customer_very_late'] = (df['arrival_delay_mins'] > 30).astype(int)

# Was the tech already running behind when they started?
df['tech_behind_flag'] = (df['tech_behind_at_start'] > 30).astype(int)
df['tech_severely_behind'] = (df['tech_behind_at_start'] > 60).astype(int)

# Did the job run over estimate?
df['job_ran_over'] = (df['duration_overrun_mins'] > 0).astype(int)
df['job_overrun_severe'] = (df['duration_overrun_mins'] > 60).astype(int)

# Appointment slot pressure — later slots accumulate more delays
df['slot_position_pct']  = ((df['appt_mins'] - 480) / 480).clip(0, 1)

# Queue efficiency — ratio of wait to actual work
df['queue_to_work_ratio'] = (df['queue_wait_mins'] / df['actual_duration_mins'].clip(lower=1)).clip(upper=5)

print(f"  Parsed and derived {len([c for c in df.columns if 'mins' in c or 'time' in c])} time features")
print(f"  Sample stats:")
print(f"    Avg arrival delay: {df['arrival_delay_mins'].mean():.1f} mins")
print(f"    Avg queue wait: {df['queue_wait_mins'].mean():.1f} mins")
print(f"    Avg tech behind at start: {df['tech_behind_at_start'].mean():.1f} mins")
print(f"    Avg duration overrun: {df['duration_overrun_mins'].mean():.1f} mins")
print(f"    Customer late rate: {df['customer_late'].mean():.1%}")
print(f"    Tech behind rate: {df['tech_behind_flag'].mean():.1%}")
print(f"    Job overrun rate: {df['job_ran_over'].mean():.1%}")

######  Encode categoricals - create labels for categorical data types
print("\n[3/8] Encoding categoricals...")
encoders = {}
for col in ['model','issue_category','service_type']:
    if col in df.columns:
        le = LabelEncoder()
        df[col+'_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        joblib.dump(le,f'./models/le_{col}.pkl')
        print(f"  {col}: {list(le.classes_)}")

######  Feature engineering
print("\n[4/8] Engineering features...")

# chronological ordering - date & appt time
df['date'] = pd.to_datetime(df['date'])

# combine date + appointment_time into one timestamp
df['appointment_dt'] = pd.to_datetime(
    df['date'].astype(str) + ' ' + df['appointment_time'].astype(str),
    errors='coerce'
)

df = df.sort_values(['date', 'appointment_dt']).reset_index(drop=True)


df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_monday'] = (df['day_of_week'] == 0).astype(int)
df['is_friday'] = (df['day_of_week'] == 4).astype(int)
df['week_of_month'] = df['date'].dt.day.apply(lambda d: (d-1)//7+1)
df['quarter'] = df['date'].dt.quarter

# Capacity features
df['slots_per_tech'] = df['service_slots'] / df['num_technicians'].clip(lower=1)
df['backlog_per_tech'] = df['backlog_size'] / df['num_technicians'].clip(lower=1)
df['backlog_per_slot'] = df['backlog_size'] / df['service_slots'].clip(lower=1)
df['appts_per_slot'] = df['appointments_per_day']  / df['service_slots'].clip(lower=1)
df['appts_per_tech'] = df['appointments_per_day']  / df['num_technicians'].clip(lower=1)
df['slot_utilisation'] = (df['appointments_per_day'] / df['service_slots'].clip(lower=1)).clip(upper=2.0)
df['pressure_index'] = df['demand_capacity_ratio'] * (1 + df['backlog_size'] / 10)

# Tesla-specific derived
df['ota_wait_est'] = df['pending_ota'] * (df['ota_size_mb'] / 20).clip(upper=60)
df['needs_charge'] = (df['battery_soc_pct'] < 20).astype(int)
df['charge_wait_est'] = np.where(df['battery_soc_pct']<20, (20-df['battery_soc_pct'])*2, 0)
df['battery_degraded'] = (df['battery_health_pct'] < 80).astype(int)
df['diag_tool_free'] = (df['diag_tools_avail'] - df['diag_tool_in_use']).clip(lower=0)
df['diag_contention'] = (df['diag_tool_in_use'] >= df['diag_tools_avail']).astype(int)
df['hv_tech_ratio'] = df['hv_certified_techs'] / df['num_technicians'].clip(lower=1)
df['calib_pressure'] = df['requires_calibration'] * \
                             (df['calibration_mins_est'] / 60).clip(upper=3)

# Interaction features
df['backlog_x_dcr'] = df['backlog_size'] * df['demand_capacity_ratio']
df['backlog_x_monday'] = df['backlog_size'] * df['is_monday']
df['complexity_x_pressure'] = df['issue_complexity'] * df['pressure_index']
df['complexity_x_techs'] = df['issue_complexity'] / df['num_technicians'].clip(lower=1)
df['parts_x_complexity'] = (1-df['parts_in_stock']) * df['issue_complexity']
df['parts_x_dcr'] = (1-df['parts_in_stock']) * df['demand_capacity_ratio']
df['ota_x_complexity'] = df['pending_ota'] * df['issue_complexity']

# Timestamp interaction features
df['late_arrival_x_pressure'] = df['customer_late'] * df['pressure_index']
df['tech_behind_x_complexity'] = df['tech_behind_at_start'] * df['issue_complexity']
df['queue_wait_x_dcr'] = df['queue_wait_mins'] * df['demand_capacity_ratio']
df['late_slot_x_backlog'] = df['slot_position_pct'] * df['backlog_size']
df['overrun_x_late_slot'] = df['duration_overrun_mins'].clip(lower=0) * df['slot_position_pct']


####### Feature list
print("\n[5/8] Building feature list...")

candidate_features = [
    # Time / date
    'month','day_of_week','is_weekend','is_monday','is_friday',
    'week_of_month','quarter',

    # Appointment time
    'appt_mins','appt_time_bucket',

    # Timestamp-derived operational features
    'arrival_delay_mins',
    'queue_wait_mins',
    'tech_behind_at_start',
    'day_running_behind_mins',
    'rolling_tech_overrun_avg',
    # 'ota_wait_mins',
    # 'charge_wait_mins',
    'slot_position_pct',
    'queue_to_work_ratio',

    # Binary time flags
    'customer_early','customer_on_time','customer_late','customer_very_late',
    'tech_behind_flag','tech_severely_behind',


    # Vehicle
    'model_enc','vehicle_age_months','mileage',

    # Job
    'issue_category_enc','service_type_enc',
    'issue_complexity','issue_duration_est',
    # 'parts_in_stock','part_lead_time_days',

    # # Tesla-specific
    'pending_ota', 'ota_size_mb', 'ota_wait_est',
    'requires_calibration', 'calibration_mins_est', 'calib_pressure',
    'battery_soc_pct', 'battery_health_pct', 'battery_thermal_event',
    'needs_charge', 'charge_wait_est', 'battery_degraded',
    'diag_tools_avail', 'diag_tool_in_use', 'diag_tool_free', 'diag_contention',
    'hv_certified_techs', 'hv_tech_ratio',

    # Capacity
    'num_technicians','service_slots','backlog_size',
    'appointments_per_day','demand_capacity_ratio',
    'slots_per_tech','backlog_per_tech','backlog_per_slot',
    'appts_per_slot','appts_per_tech','slot_utilisation',

    # Interactions
    'backlog_x_dcr','backlog_x_monday',
    'late_arrival_x_pressure','tech_behind_x_complexity',
    'queue_wait_x_dcr','late_slot_x_backlog','overrun_x_late_slot',
]
# 'ota_x_complexity',    'complexity_x_pressure','complexity_x_techs',
#     'parts_x_complexity','parts_x_dcr'

features = [f for f in candidate_features if f in df.columns]
missing  = [f for f in candidate_features if f not in df.columns]
print(f"  Using {len(features)} features")
if missing:
    print(f"  Not found in dataset (skipped): {missing}")

joblib.dump(features, './models/feature_list.pkl')

####### Prepare X / y
print("\n[6/8] Preparing training data...")
X = df[features].copy().fillna(df[features].median(numeric_only=True))
y_class = df['delay_risk']
y_reg = df['wait_time']

X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_class, test_size=0.2, stratify=y_class,
    shuffle=True, random_state=42
)
_, _, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, shuffle=True, random_state=42
)

neg = (y_train_class==0).sum()
pos = (y_train_class==1).sum()
scale = round(neg/max(pos,1), 2)
print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")
print(f"  Delay rate — train: {y_train_class.mean():.3f}  test: {y_test_class.mean():.3f}")
print(f"  scale_pos_weight: {scale}")

###### Train all models
print("\n[7/8] Training models...")

classifiers = {
    'LogisticRegression': LogisticRegression(
        max_iter=2000, class_weight='balanced', random_state=42),
    'RandomForest': RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale, eval_metric='auc',
        random_state=42, verbosity=0, n_jobs=-1),
    'LightGBM': LGBMClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, verbosity=-1, n_jobs=-1),
    'CatBoost': CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        auto_class_weights='Balanced', eval_metric='AUC',
        random_state=42, verbose=0),
}

regressors = {
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1),
    'XGBoost': XGBRegressor(
        n_estimators=500, max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        n_jobs=-1),
    'LightGBM': LGBMRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        n_jobs=-1),
    'CatBoost': CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        random_state=42,
        verbose=0),
}

clf_results = {}
clf_models  = {}
for name, model in classifiers.items():
    print(f"  [CLF] {name}...", end=" ", flush=True)
    t = time.time()
    model.fit(X_train, y_train_class)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    clf_results[name] = {
        'AUC': round(roc_auc_score(y_test_class, y_proba), 4),
        'F1': round(f1_score(y_test_class, y_pred), 4),
        'Recall': round(recall_score(y_test_class, y_pred), 4),
        'Prec': round(precision_score(y_test_class, y_pred), 4),
        'Secs': round(time.time()-t, 1)
    }
    clf_models[name] = model
    joblib.dump(model, f'./models/all/clf_{name.lower()}.pkl')
    r = clf_results[name]
    print(f"AUC={r['AUC']}  F1={r['F1']}  Recall={r['Recall']}  ({r['Secs']}s)")

reg_results = {}
reg_models  = {}
for name, model in regressors.items():
    print(f"  [REG] {name}...", end=" ", flush=True)
    t = time.time()
    model.fit(X_train, y_train_reg)
    y_pred = model.predict(X_test)
    reg_results[name] = {
        'MAE': round(mean_absolute_error(y_test_reg, y_pred), 2),
        'R2': round(r2_score(y_test_reg, y_pred), 4),
        'MAPE': round(mean_absolute_percentage_error(y_test_reg, y_pred), 4),
        'Secs': round(time.time()-t, 1)
    }
    reg_models[name] = model
    joblib.dump(model, f'./models/all/reg_{name.lower()}.pkl')
    r = reg_results[name]
    print(f"MAE={r['MAE']}m  R2={r['R2']}  ({r['Secs']}s)")

# Quantile models
print("  [REG] LightGBM quantiles (p10/p50/p90)...", end=" ", flush=True)
q_models = {}
for q, label in [(0.10,'p10'),(0.50,'p50'),(0.90,'p90')]:
    qm = LGBMRegressor(objective='quantile', alpha=q, n_estimators=300,
                        max_depth=6, learning_rate=0.05,
                        random_state=42, verbosity=-1)
    qm.fit(X_train, y_train_reg)
    q_models[label] = qm
    joblib.dump(qm, f'./models/quantile_{label}.pkl')
p10p = q_models['p10'].predict(X_test)
p90p = q_models['p90'].predict(X_test)
cov  = ((y_test_reg >= p10p) & (y_test_reg <= p90p)).mean()
print(f"80% interval coverage: {cov:.3f}")

######  Pick best, threshold, save
print("\n[8/8] Saving best models...")

clf_df = pd.DataFrame(clf_results).T.sort_values('AUC', ascending=False)
reg_df = pd.DataFrame(reg_results).T.sort_values('MAE')
print("\n  Classification results:")
print(clf_df.to_string())
print("\n  Regression results:")
print(reg_df.to_string())

best_clf_name = clf_df['AUC'].idxmax()
best_reg_name = reg_df['MAE'].idxmin()
best_clf = clf_models[best_clf_name]
best_reg = reg_models[best_reg_name]

# Threshold tuning
y_proba_best  = best_clf.predict_proba(X_test)[:,1]
thresh_rows   = []
for t in np.arange(0.10, 0.91, 0.05):
    yp = (y_proba_best >= t).astype(int)
    if yp.sum()==0 or (1-yp).sum()==0: continue
    thresh_rows.append({
        'Threshold': round(t,2),
        'Recall': round(recall_score(y_test_class, yp), 4),
        'Precision': round(precision_score(y_test_class, yp), 4),
        'F1': round(f1_score(y_test_class, yp), 4),
    })
thresh_df    = pd.DataFrame(thresh_rows)
good         = thresh_df[thresh_df['Recall'] >= 0.80]
best_thresh  = good.loc[good['F1'].idxmax(),'Threshold'] if len(good) else 0.50
print(f"\n  Threshold tuning (best for 80%+ recall): {best_thresh}")
print(thresh_df.to_string(index=False))

# Save
joblib.dump(best_clf,'./models/rf_model_classification_delay_risk.pkl')
joblib.dump(best_reg,'./models/rf_model_regression_wait_time.pkl')
joblib.dump(best_thresh, './models/classification_threshold.pkl')
joblib.dump(features,'./models/feature_list.pkl')
joblib.dump({
    'classifier_name': best_clf_name,
    'regressor_name': best_reg_name,
    'threshold': best_thresh,
    'features': features,
    'clf_auc': clf_df.loc[best_clf_name,'AUC'],
    'clf_f1': clf_df.loc[best_clf_name,'F1'],
    'reg_mae': reg_df.loc[best_reg_name,'MAE'],
    'reg_r2': reg_df.loc[best_reg_name,'R2'],
    'train_size': len(X_train),
    'test_size': len(X_test),
    'timestamp_features_included': True,
}, './models/model_metadata.pkl')

print("-----------------------------------------------------------------------------------------------------------")
# print(f"\n  \033[1m"+ "Best classifier : '{best_clf_name}'  (AUC='{clf_df.loc[best_clf_name,'AUC']}')" + "\033[0m")
print(f"\n\033[1mBest classifier : {best_clf_name} (AUC={clf_df.loc[best_clf_name, 'AUC']})\033[0m")
print(f"  Best regressor  : {best_reg_name}  (MAE={reg_df.loc[best_reg_name,'MAE']}m)")
print(f"  Threshold       : {best_thresh}")
print("-----------------------------------------------------------------------------------------------------------")

if hasattr(best_clf,'feature_importances_'):
    fi = pd.DataFrame({'Feature':features,
                       'Importance':best_clf.feature_importances_}
                      ).sort_values('Importance',ascending=False).head(20)
    print(f"\n  Top 20 features ({best_clf_name}):")
    print(fi.to_string(index=False))

print("\nDone.")