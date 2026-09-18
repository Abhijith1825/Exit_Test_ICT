import os
from typing import Dict, List

import pandas as pd
from flask import Flask, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "G1.csv")
TARGET_COLUMN = "Best_Shopping_Mode"
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Occupation_Status",
    "Country_Region",
    "Monthly_Income_Range",
    "Shop_Most_Where",
    "Online_Shopping_Frequency",
    "Preferred_Platform",
    "Primary_Device",
    "Top_Spending_Category",
    "Review_Influence_Score",
    "Social_Ads_Influence_Score",
    "Price_Comparison_Frequency",
    "Discount_Importance_Score",
    "Follows_Brands_On_Social",
    "Likely_To_Buy_After_Ad_Score",
    "Makes_Shopping_List",
    "Impulse_Purchase_Frequency",
    "Return_Frequency",
    "Avg_Monthly_Spend_NonEssentials",
    "Preferred_Payment_Method",
    "Uses_BNPL_Installments",
    "Online_Shopping_Satisfaction_Score",
    "Trust_In_Online_Reviews_Score",
    "Regret_After_Purchase_Frequency",
    "Reason_Prefer_Online",
    "Reason_Prefer_InStore",
]

NUMERIC_FIELDS = {
    "Age",
    "Review_Influence_Score",
    "Social_Ads_Influence_Score",
    "Price_Comparison_Frequency",
    "Discount_Importance_Score",
    "Likely_To_Buy_After_Ad_Score",
    "Online_Shopping_Satisfaction_Score",
    "Trust_In_Online_Reviews_Score",
    "Regret_After_Purchase_Frequency",
}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    for column in FEATURE_COLUMNS:
        if column in df.columns:
            df[column] = df[column].replace(["", " ", "nan", "NaN", "N/A", "n/a"], pd.NA)

    return df


def train_model():
    df = load_dataset()
    working_df = df.loc[:, FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    working_df = working_df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    for col in working_df.columns:
        if col in NUMERIC_FIELDS:
            working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    X = working_df[FEATURE_COLUMNS]
    y = working_df[TARGET_COLUMN]

    numeric_features = [col for col in FEATURE_COLUMNS if col in NUMERIC_FIELDS]
    categorical_features = [col for col in FEATURE_COLUMNS if col not in NUMERIC_FIELDS]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return {
        "pipeline": pipeline,
        "accuracy": round(float(accuracy), 4),
        "target_distribution": y.value_counts().to_dict(),
        "dataset_summary": {
            "total_rows": int(len(working_df)),
            "feature_count": len(FEATURE_COLUMNS),
            "target_column": TARGET_COLUMN,
        },
    }


MODEL_STATE = train_model()


def get_form_fields() -> List[Dict[str, object]]:
    df = load_dataset()
    fields: List[Dict[str, object]] = []

    for field in FEATURE_COLUMNS:
        if field in NUMERIC_FIELDS:
            fields.append(
                {
                    "name": field,
                    "label": field.replace("_", " "),
                    "type": "number",
                    "min": 0,
                    "max": 100,
                    "step": 1,
                }
            )
        else:
            options = [
                str(value)
                for value in sorted(df[field].dropna().unique().tolist())
                if str(value).strip() != ""
            ]
            fields.append(
                {
                    "name": field,
                    "label": field.replace("_", " "),
                    "type": "select",
                    "options": options,
                }
            )

    return fields


@app.route("/")
def home():
    target_distribution = MODEL_STATE["target_distribution"]
    total_rows = MODEL_STATE["dataset_summary"]["total_rows"]

    return render_template(
        "index.html",
        summary=MODEL_STATE["dataset_summary"],
        model_accuracy=MODEL_STATE["accuracy"],
        target_distribution=target_distribution,
        total_rows=total_rows,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    form_fields = get_form_fields()
    prediction = None
    form_values = {}

    if request.method == "POST":
        input_row = {}
        for field in FEATURE_COLUMNS:
            value = request.form.get(field, "")
            if value is None or value == "":
                input_row[field] = None
            elif field in NUMERIC_FIELDS:
                try:
                    input_row[field] = float(value)
                except ValueError:
                    input_row[field] = None
            else:
                input_row[field] = value

        form_values = input_row

        feature_frame = pd.DataFrame([input_row], columns=FEATURE_COLUMNS)
        prediction = MODEL_STATE["pipeline"].predict(feature_frame)[0]

    return render_template(
        "predict.html",
        form_fields=form_fields,
        prediction=prediction,
        form_values=form_values,
    )


if __name__ == "__main__":
    app.run(debug=True)
