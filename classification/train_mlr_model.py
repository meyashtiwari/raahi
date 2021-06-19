import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Loading data
file_path = os.path.join('..', 'data', 'compiled_node_dataset.csv')
data_nodes = pd.read_csv(file_path)

x = data_nodes[['Battery', 'Internal']].values
y = data_nodes['Class'].values

ml = LogisticRegression()
ml.fit(x, y)

# Save Model
save_model = os.path.join('..', 'models', 'mlr_model.pkl')
joblib.dump(ml, save_model)