import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

DATA_FILE = "ai4i2020.csv" # Update this path if your dataset is located elsewhere

# 1. Load the data
print("Loading data...")
df = pd.read_csv(DATA_FILE)

# 2. Clean the Kaggle Dataset
# Check if it's the real Kaggle dataset by looking for 'Machine failure'
if 'Machine failure' in df.columns:
    print("Cleaning real Kaggle dataset columns...")
    
    # Drop ID columns and specific failure modes to prevent data leakage
    cols_to_drop = ['UDI', 'Product ID', 'Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Rename columns so they have no spaces/brackets (Matches our FastAPI app)
    df = df.rename(columns={
        'Air temperature [K]': 'air_temp',
        'Process temperature [K]': 'process_temp',
        'Rotational speed [rpm]': 'rotational_speed',
        'Torque [Nm]': 'torque',
        'Tool wear [min]': 'tool_wear',
        'Machine failure': 'failure'
    })

# 3. Separate features (X) and target (y)
X = df.drop("failure", axis=1)
y = df["failure"]

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Handle class imbalance with SMOTE
print("Applying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)  # type: ignore

# 6. Train the Random Forest Model
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# 7. Save the trained model
MODEL_FILE = "xgboost_model.pkl" # Keeping this name so we don't have to change app.py!
joblib.dump(model, MODEL_FILE)
print(f"Success! Model trained and saved as {MODEL_FILE}")