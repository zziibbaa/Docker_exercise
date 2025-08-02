import os
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

import mlflow

mlflow.set_tracking_uri("http://localhost:5555")
print("Tracking URI:", mlflow.get_tracking_uri())

# حذف experiment اضافی برای اطمینان
mlflow.set_experiment("Default")  # یا همون که در UI بود

with mlflow.start_run():
    mlflow.log_param("param1", 42)
    mlflow.log_metric("rmse", 0.123)
    print("✅ Run logged successfully")