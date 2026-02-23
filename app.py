from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("power_model.pkl")
scaler = joblib.load("scaler.pkl")

prediction_history = []
climate_history = []

AVERAGE_CONSUMPTION = 95000 # You can adjust based on dataset mean


def get_climate_label(temp, hour):
    if 6 <= hour < 18:
        if temp >= 30:
            return "Hot Day ☀️"
        elif temp <= 12:
            return "Cold Day ❄️"
        else:
            return "Mild Day 🌤️"
    else:
        if temp >= 30:
            return "Hot Night 🌙"
        elif temp <= 12:
            return "Cold Night 🌌"
        else:
            return "Mild Night 🌙"



@app.route("/")
def home():
    return render_template("index.html",
                           history=prediction_history,
                           climates=climate_history)


@app.route("/predict", methods=["POST"])
def predict():
    global prediction_history, climate_history

    try:
        temp = float(request.form["Temperature"])
        humidity = float(request.form["Humidity"])
        wind = float(request.form["WindSpeed"])
        general_diffuse = float(request.form["GeneralDiffuseFlows"])
        diffuse = float(request.form["DiffuseFlows"])
        hour = int(request.form["Hour"])
        day = int(request.form["Day"])
        month = int(request.form["Month"])
        day_of_week = int(request.form["DayOfWeek"])
        weekend = int(request.form["Weekend"])
        rolling_mean = float(request.form["Rolling_Mean"])
    except ValueError:
        return render_template("index.html",
                               history=prediction_history,
                               climates=climate_history,
                               prediction_text="⚠️ Invalid input! Please enter numbers only.")

    features = [[temp, humidity, wind, general_diffuse, diffuse,
                 hour, day, month, day_of_week, weekend, rolling_mean]]

    scaled_features = scaler.transform(features)
    base_prediction = model.predict(scaled_features)[0]

    variation = (temp * 120) - (humidity * 30) + (hour * 80)
    prediction = base_prediction + variation
    prediction = round(prediction, 2)

    prediction_history.append(prediction)

    climate_label = get_climate_label(temp, hour)
    climate_history.append(climate_label)

    return render_template("result.html",
                           prediction_text=f"Predicted Total Power Consumption: {prediction}",
                           prediction_value=prediction,
                           avg_value=AVERAGE_CONSUMPTION,
                           history=prediction_history,
                           climates=climate_history)


if __name__ == "__main__":
    app.run(debug=True)