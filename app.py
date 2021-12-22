from flask import Flask, jsonify, request, render_template
import joblib
import numpy as np

# Intial change
# second change
# 3rd change

model = joblib.load("pipe_NB.pkl")

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['message']
    
        res = model.predict([text])[0]
        if res == 1:
            res = "The mail is spam"
        else:
            res = "The mail is ham"
        
        return render_template('index.html', prediction_text = res)
    
if __name__ == "__main__":
    app.run(debug=True)
