import os
import pandas as pd
from sklearn import metrics
import joblib
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Loading data
file_path = os.path.join('..', 'data', 'test_dataset.csv')
data_nodes = pd.read_csv(file_path)

x = data_nodes[['Class', 'Range']].values
y = data_nodes['Network_type'].values

model_path = os.path.join('..', 'models', 'network_prediction_model.pkl')
dtc = joblib.load(model_path)

y_pred = dtc.predict(x)

print("Accuracy:",metrics.accuracy_score(y, y_pred))
print("\nScore:",dtc.score(x, y))

plt.figure(figsize=(10,10))
plt.scatter(y, y_pred, color='red')
plt.plot(['MANET', 'DTN', 'Bluetooth'])
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Network Prediction using Decision Tree Classifier')