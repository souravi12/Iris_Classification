from flask import Flask, request, render_template
import numpy as np
import pickle
app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])

def predict():
  
    features = [float(x) for x in request.form.values()]
    
  
    final_features = np.array(features).reshape(1, -1)
    

    prediction = model.predict(final_features)
    output = prediction[0]
    
    iris_classes = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    flower_name = iris_classes[output]
  
    return render_template('index.html', prediction_text=f'The predicted iris flower is: {flower_name}')

if __name__ == '__main__':
    app.run(debug=True)
