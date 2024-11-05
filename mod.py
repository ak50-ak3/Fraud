
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess your dataset
df = pd.read_csv("Synthetic_Financial_datasets_log.csv")
df.drop_duplicates(inplace=True)

# Ensure you are using the right label encoding
label_encoder = LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])
df = df.drop(columns=['nameOrig', 'nameDest'])

# Define features and target
X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df['isFraud']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "/Users/admin/Desktop/FOMLPROJECT/decision_tree_fraud.pkl")

# Save the label encoder
joblib.dump(label_encoder, "/Users/admin/Desktop/FOMLPROJECT/label_encoder.pkl")
