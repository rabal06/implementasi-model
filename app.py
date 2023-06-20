import csv
import numpy as np
from flask import request
from flask import Flask, render_template
import joblib

app = Flask(__name__)

model = joblib.load("earthquake_dtc.pkl")

@app.route("/", methods=['GET'])
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        results = []
        magnitude = float(request.form["magnitude"])
        alert = int(request.form["alert"])
        prediction = model.predict([[magnitude, alert]])
        for row in model:
            if row[0] == str(prediction[0]):
                results.append({
                    'title': row[1],
                    'location': row[2],
                    'continent': row[3]
                })

        # membaca earthquake data dari file CSV 
        with open('earthquake_data.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter = ',')
                csvwriter.writerow(["Title", "Location", "Continent"])
                for r in results:
                    csvwriter.writerow([r["title"], r["location"], r["continent"]])

        # menampilkan hasil prediksi pada halaman result.html
        return render_template('/templates/result.html', results=results)
        
        
    else:
        # menghandle GET request
        return render_template('/templates/index.html')

if __name__ == "__main__":
    app.run(debug=True)