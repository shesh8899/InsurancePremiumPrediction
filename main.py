import os
import pandas as pd
from xgboost import XGBRegressor
from joblib import load as ld


# ========= Paths =========
base_path = "/Users/shashankpatil/Desktop"
param_path = os.path.join(base_path, "model1.joblib")
model_path = os.path.join(base_path, "model.json")


# ========= Load artifacts (as saved by your notebook) =========
param = ld(param_path)  # keys in your notebook: 'scaler', 'dummycols', 'medical_score_min_max', 'allcols'
scaler = param.get("scaler", None)
if scaler is None:
    raise KeyError(f"'scaler' not found in joblib. Available keys: {list(param.keys())}.")

DUMMY_COLS = list(param.get("dummycols", []))

MEDI_MIN, MEDI_MAX = None, None
if isinstance(param.get("medical_score_min_max"), (list, tuple)) and len(param["medical_score_min_max"]) == 2:
    MEDI_MIN, MEDI_MAX = param["medical_score_min_max"]

# Load trained XGBoost model and get its exact expected feature names
xgb = XGBRegressor()
xgb.load_model(model_path)
model_features = xgb.get_booster().feature_names
if not model_features:
    # Fallback if model lacks names (rare) — use scaler features then remove known dropped ones.
    if hasattr(scaler, "feature_names_in_"):
        model_features = list(scaler.feature_names_in_)
    else:
        model_features = list(param.get("allcols", []))
    # Remove features dropped before training (per notebook cell that dropped them)
    for col in ["total_score", "Smoking_Status_Regular"]:
        if col in model_features:
            model_features.remove(col)

# Scaler expects exactly the features it was fit on
if hasattr(scaler, "feature_names_in_"):
    scaler_features = list(scaler.feature_names_in_)
else:
    scaler_features = list(param.get("allcols", []))
    if not scaler_features:
        raise KeyError(
            "Could not infer scaler feature names. Expected 'scaler.feature_names_in_' or 'allcols' in joblib."
        )


# ========= Input container =========
class input:
    def __init__(
        self,
        age, Gender, Region, Marital_Status, Number_of_Dependants,
        BMI_Category, Smoking_Status, Employment_Status, Income_Level,
        Income_Lakhs, Medical_History, Insurance_Plan, Genetical_Risk
    ):
        self.age = age
        self.Gender = Gender
        self.Region = Region
        self.Marital_Status = Marital_Status
        self.Number_of_Dependants = Number_of_Dependants
        self.BMI_Category = BMI_Category
        self.Smoking_Status = Smoking_Status
        self.Employment_Status = Employment_Status
        self.Income_Level = Income_Level
        self.Income_Lakhs = Income_Lakhs
        self.Medical_History = Medical_History
        self.Insurance_Plan = Insurance_Plan
        self.Genetical_Risk = Genetical_Risk


# ========= Model wrapper =========
class model(input):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Recreate LabelEncoder behavior deterministically (alphabetical class order)
    def _encode_ordinals_like_labelencoder(self, df: pd.DataFrame) -> pd.DataFrame:
        # Insurance_Plan (classes from your UI)
        plan_classes = ["Bronze", "Gold", "Silver"]
        plan_mapping = {k: i for i, k in enumerate(sorted(plan_classes))}
        if "Insurance_Plan" in df.columns:
            df["Insurance_Plan"] = df["Insurance_Plan"].map(plan_mapping).fillna(0)

        # Income_Level (classes from your UI)
        income_classes = ["> 40L", "<10L", "10L - 25L", "25L - 40L"]
        income_mapping = {k: i for i, k in enumerate(sorted(income_classes))}
        if "Income_Level" in df.columns:
            df["Income_Level"] = df["Income_Level"].map(income_mapping).fillna(0)

        return df

    def _canonicalize_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Light canonicalization for categories to match training text exactly."""
        # Normalize a few known variants
        if "Smoking_Status" in df.columns:
            val = str(df.loc[0, "Smoking_Status"]).strip().lower()
            if val in {"no smoking", "no-smoking", "nosmoking", "no  smoking", "no  - smoking", "no  _ smoking", "no  / smoking"}:
                df.loc[0, "Smoking_Status"] = "No Smoking"
            elif val in {"regular"}:
                df.loc[0, "Smoking_Status"] = "Regular"
            elif val in {"occasional"}:
                df.loc[0, "Smoking_Status"] = "Occasional"
        return df

    def _medical_history_score(self, s: pd.Series) -> pd.Series:
        """
        Compute normalized risk score from Medical_History.
        Uses notebook's saved min/max if available; otherwise divides by 14.
        Always returns a Series.
        """
        s = s.astype(str).str.lower()
        parts = s.str.split(r"\s*&\s*")
        d1 = parts.str.get(0).str.strip()
        d2 = parts.str.get(1).fillna("none").str.strip()

        risk_scores = {
            "diabetes": 6,
            "heart disease": 8,
            "high blood pressure": 6,
            "thyroid": 5,
            "no disease": 0,
            "none": 0,
        }
        score1 = d1.map(risk_scores).fillna(0)
        score2 = d2.map(risk_scores).fillna(0)
        total = score1 + score2

        if MEDI_MIN is not None and MEDI_MAX is not None and MEDI_MAX != MEDI_MIN:
            norm = (total - MEDI_MIN) / (MEDI_MAX - MEDI_MIN)
        else:
            norm = total / 14.0

        return norm.clip(lower=0, upper=1)

    def dataframecreation(self):
        # ---------- Build the base row (KEEP training-time column names) ----------
        df = pd.DataFrame({
            "Age": [self.age],
            "Gender": [self.Gender],
            "Region": [self.Region],
            "Marital_Status": [self.Marital_Status],
            "Number_of_Dependants": [self.Number_of_Dependants],
            "BMI_Category": [self.BMI_Category],
            "Smoking_Status": [self.Smoking_Status],
            "Employment_Status": [self.Employment_Status],
            "Income_Lakhs": [self.Income_Lakhs],
            "Medical_History": [self.Medical_History],
            "Insurance_Plan": [self.Insurance_Plan],
            "Genetical_Risk": [self.Genetical_Risk],
            "Income_Level": [self.Income_Level],
        })

        # Canonicalize category text (e.g., 'NO Smoking' -> 'No Smoking')
        df = self._canonicalize_inputs(df)

        # Align training spellings:
        # 1) Marital_Status -> Marital_status
        if "Marital_status" in DUMMY_COLS and "Marital_Status" in df.columns:
            df.rename(columns={"Marital_Status": "Marital_status"}, inplace=True)
        # 2) Number_of_Dependants -> Number Of Dependants
        if "Number Of Dependants" in scaler_features and "Number_of_Dependants" in df.columns:
            df.rename(columns={"Number_of_Dependants": "Number Of Dependants"}, inplace=True)

        # Encode the two LabelEncoded features to match training
        df = self._encode_ordinals_like_labelencoder(df)

        # Add normalized medical risk feature
        df["normalized_score"] = self._medical_history_score(df["Medical_History"])

        # One-hot encode ONLY the columns used during training
        existing = [c for c in DUMMY_COLS if c in df.columns]
        if existing:
            dummies = pd.get_dummies(df[existing], drop_first=True)
            df.drop(columns=existing, inplace=True)
            if not dummies.empty:
                df = pd.concat([df, dummies], axis=1)

        # Remove raw columns that were not part of the model
        if "Medical_History" not in scaler_features and "Medical_History" in df.columns:
            df.drop(columns=["Medical_History"], inplace=True, errors="ignore")
        # Explicitly drop training-time removed columns if they sneak in
        for col_to_remove in ["Smoking_Status_Regular", "total_score"]:
            if col_to_remove in df.columns:
                df.drop(columns=[col_to_remove], inplace=True, errors="ignore")

        # ---------- CRITICAL: make scaler happy first ----------
        # 1) Reindex to the scaler's expected feature set (fill missing with 0)
        df_for_scaler = df.reindex(columns=scaler_features, fill_value=0)
        # 2) Scale
        df_scaled = pd.DataFrame(scaler.transform(df_for_scaler), columns=scaler_features)

        # ---------- Now make model happy (exact columns/order) ----------
        # 3) Select exactly model_features, in order; fill any missing (edge-case) with 0
        #    (This handles cases where scaler was fit before a late feature drop.)
        df_model = df_scaled.reindex(columns=model_features, fill_value=0)

        # Predict with XGBoost using the model-aligned frame
        return xgb.predict(df_model)