from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

# Load your data from the CSV file
data = pd.read_csv('balance-scale.csv', delimiter=',', header=0)

# Assuming your target variable is 'Class' and features are the remaining columns
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MLPClassifier (Artificial Neural Network)
ann_model = MLPClassifier(
    hidden_layer_sizes=(100,),  # Adjust the number of neurons in the hidden layer
    max_iter=1000,               # Increase max_iter
    random_state=42
)

ann_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_ann = ann_model.predict(X_test_scaled)

# Evaluate the MLPClassifier
accuracy_ann = accuracy_score(y_test, y_pred_ann)
print(f"ANN Accuracy: {accuracy_ann:.2f}")

# Support Vector Classifier (SVC)
svc_model = SVC()

svc_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_svc = svc_model.predict(X_test_scaled)

# Evaluate the SVC
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print(f"SVC Accuracy: {accuracy_svc:.2f}")
