import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('rainfall_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature engineering function (from your code)
def engineer_features(input_data):
    df = pd.DataFrame([input_data])
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['humidity_dewpoint'] = df['humidity'] * df['dewpoint']
    df['pressure_change'] = 0  # No previous day data
    df['humidity_cloud'] = df['humidity'] * df['cloud']
    df['prev_rainfall'] = 0  # No previous day
    df['wind_interaction'] = df['windspeed'] * df['winddirection']
    df['sunshine_cloud'] = df['sunshine'] * df['cloud']
    df['temp_humidity'] = df['temparature'] * df['humidity']
    df['dewpoint_temp_diff'] = df['temparature'] - df['dewpoint']
    df['pressure_humidity'] = df['pressure'] * df['humidity']
    
    features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud',
                'sunshine', 'winddirection', 'windspeed', 'temp_range', 'humidity_dewpoint',
                'pressure_change', 'humidity_cloud', 'prev_rainfall', 'wind_interaction',
                'sunshine_cloud', 'temp_humidity', 'dewpoint_temp_diff', 'pressure_humidity']
    X = scaler.transform(df[features])
    return X

# Prediction function
def predict_rainfall(input_data):
    X = engineer_features(input_data)
    prediction = model.predict(X)[0]
    return "yes" if prediction == 1 else "no"

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    image = 'cloudy.jpg'  # Default image
    prediction = None
    
    if request.method == 'POST':
        # Get form data
        try:
            input_data = {
                'pressure': float(request.form['pressure']),
                'maxtemp': float(request.form['maxtemp']),
                'temparature': float(request.form['temparature']),
                'mintemp': float(request.form['mintemp']),
                'dewpoint': float(request.form['dewpoint']),
                'humidity': float(request.form['humidity']),
                'cloud': float(request.form['cloud']),
                'sunshine': float(request.form['sunshine']),
                'winddirection': float(request.form['winddirection']),
                'windspeed': float(request.form['windspeed'])
            }
            # Predict rainfall
            prediction = predict_rainfall(input_data)
            image = 'rainy.jpg' if prediction == 'yes' else 'sunny.jpg'
        except ValueError:
            prediction = "Error: Please enter valid numbers"
    
    return render_template('index.html', image=image, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)