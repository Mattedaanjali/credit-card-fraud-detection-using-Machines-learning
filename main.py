import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

#load dataset
df = pd.read_csv("creditcard.csv")

#display 5 rows
print(df.head())
#display dataset shape
print(df.shape)
#displays datset information
print(df.info())
#check missing values 
print(df.isnull().sum())
#stastical summary
print(df.describe())
#correlation matrix
print(df.corr())
#features(independent variables)
X = df.drop('Class', axis=1)

#targets(dependent variables)
y = df['Class']
print(X.head())
print(y.head())


#Train test split
X_train, X_test, y_train, y_test= train_test_split(
    X,
    y,
    test_size =0.2,
    random_state=42
    
)
print("\n Train Test split")
print("X Train Shape:", X_train.shape)
print("X Test Shape:", X_test.shape)
print("Y train shape:", y_train.shape)
print("Y test shape:", y_test. shape)

#Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("\n feature scaling complted successfully")
print("\n first 5 scaler trainig records")
print(X_train[:5])

#logistic regreesion
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
print("model training completed successfully")
y_pred = model.predict(X_test)
print("\n first 10 predictions:")
print(y_pred[:10])
#calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model:", accuracy)
#classification report
report = classification_report(y_test, y_pred)
print("Classification report")
print(report)
#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#roc curve
y_proba = model.predict_proba(X_test)[:,1]
#calculate AUC 
auc = roc_auc_score(y_test, y_proba)

print("AUC Score:", auc)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0,1],[0,1],'r--')

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
joblib.dump(model, "fraud_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel and scaler Saved Successfully!")


plt.show()

print(df['Class'].value_counts())
df['Class'].value_counts().plot(kind='bar')

plt.title("Distribution of Genuine and Fraud Transactions")
plt.xlabel("Class")
plt.ylabel("Count")

plt.show()
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
#SAVE MODEL
#joblib.dump(model, "fraud_detection_model.pkl")

#print("\nModel Saved Successfully!")
