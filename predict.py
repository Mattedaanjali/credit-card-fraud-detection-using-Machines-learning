import joblib
import numpy as np

# Load Saved Model
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

print("Model Loaded Successfully!")
#create sample trasaction
sample_transaction = np.zeros((1, 30))
#scaler transaction
sample_transaction =scaler.transform(sample_transaction)

#predict
prediction = model.predict(sample_transaction)
if prediction[0] ==0:
      print("genuine transaction")
else:
    print("fraud transaction")
