from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load and preprocess the data
df = pd.read_excel('02_payment_20170101_20191231.xltx')
df['update'] = pd.to_datetime(df['update'], errors='coerce')
df['BILLDATE'] = pd.to_datetime(df['BILLDATE'], errors='coerce')
df['days_diff'] = (df['update'] - df['BILLDATE']).dt.days
df['on_time'] = (df['days_diff'] <= 0).astype(int)

X = df[['days_diff']]
y = df['on_time']
X = X.fillna(X.mean())
X = X.copy()
X.fillna(X.mean(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

forest_clf = RandomForestClassifier()
forest_clf.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    PDate = request.form['PDate']
    BillDate = request.form['BillDate']

    PDate = pd.to_datetime(PDate)
    BillDate = pd.to_datetime(BillDate)
    days_diff = (PDate - BillDate).days

    data = pd.DataFrame({'days_diff': [days_diff]})
    data_scaled = scaler.transform(data)

    prediction = forest_clf.predict(data_scaled)
    result = 'Lancar' if prediction[0] == 1 else 'Tertunggak'

    return jsonify({'prediction': result})

@app.route('/get_percentage', methods=['GET'])
def get_percentage():
    total = len(y)
    on_time_count = sum(y)
    late_count = total - on_time_count
    
    on_time_percentage = (on_time_count / total) * 100
    late_percentage = (late_count / total) * 100
    
    return jsonify({
        'on_time': on_time_percentage,
        'late': late_percentage
    })

if __name__ == '__main__':
    app.run(debug=True, port=5002)
