import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn


# ======================
# ARGUMENT
# ======================
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="preprocessed_data.csv")
args = parser.parse_args()


# ======================
# MLFLOW SETUP
# ======================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Student_Performance_Model")

mlflow.autolog()


# ======================
# LOAD DATA
# ======================
df = pd.read_csv(args.data_path)

X = df.drop(columns=["Exam_Score"])
y = df["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ======================
# TRAIN MODEL (TANPA start_run)
# ======================
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R2 Score:", r2)

# manual logging tambahan
mlflow.log_metric("MAE", mae)
mlflow.log_metric("R2", r2)

# simpan model
mlflow.sklearn.log_model(model, "model")
