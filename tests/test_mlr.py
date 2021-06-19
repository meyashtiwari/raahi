import os
import pandas as pd
from sklearn.metrics import r2_score
import joblib
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Loading data
file_path = os.path.join('..', 'data', 'test_dataset.csv')
data_nodes = pd.read_csv(file_path)

x = data_nodes[['Battery', 'Internal']].values
y = data_nodes['Class'].values

model_path = os.path.join('..', 'models', 'mlr_model.pkl')
ml = joblib.load(model_path)

y_pred = ml.predict(x)

print("\nAccuracy:",ml.score(x, y))
print("\nAUC/ROC Score:",r2_score(y, y_pred))

plt.figure(figsize=(10,10))
plt.scatter(y, y_pred, color='red')
plt.plot([1,2,3,4], [1,2,3,4])
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Class Prediction using MLR Model')

pred_y_df = pd.DataFrame({'Actual Value':y, 'Predicted Value':y_pred, 'Difference':y-y_pred})
pred_y_df