from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
model = load_model('model/ams_prediction_model.keras')

def normalize_input(features):
    """Normalize input features using original min-max scaling"""
    feature_min = np.array([2850, 0, 0, 18, 0, 0, 0, 0])
    feature_max = np.array([4559, 1, 1, 77, 1, 1, 8.607, 1])
    
    ranges = feature_max - feature_min
    ranges[ranges == 0] = 1  # prevent division by zero
    
    normalized = (features - feature_min) / ranges
    return normalized

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            features = np.array([
                    float(request.form['ort']),
                    float(request.form['guide']),
                    float(request.form['gender']),
                    float(request.form['age']),
                    float(request.form['slow']),
                    float(request.form['pre_acclimatization']),
                    float(request.form['knowledge_score']),
                    float(request.form['ams_history'])
                ], dtype=np.float32)
            
            normalized_features = normalize_input(features).reshape(1, -1)
            probability = float(model.predict(normalized_features)[0][0])
            result = "High AMS Risk" if probability > 0.5 else "Low AMS Risk"
            
            return render_template('index.html',
                                probability=f"{probability:.2%}",
                                show_result=True)
            
        except Exception as e:
            return render_template('index.html',
                                error=f"Invalid input: {str(e)}",
                                show_result=False)
    
    return render_template('index.html', show_result=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))  # Removed debug=True