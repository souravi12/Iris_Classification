from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save new model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model retrained and saved as model.pkl ✅")
