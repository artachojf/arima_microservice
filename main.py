from flask import Flask, jsonify, request
import pandas as pd
from pmdarima.arima import auto_arima
import os

app = Flask(__name__)

@app.route('/predictValue', methods=['POST'])
def predict_value():
    json = request.get_json()
    column = pd.DataFrame(json['column'], columns=['estimation'])
    n = json['n']
    return jsonify({'prediction': get_prediction(column, n)})

def get_prediction(column: pd.DataFrame, nPeriods: int) -> float:
    model = auto_arima(column, start_p=0, start_q=0,
                    test='adf',
                    max_p=5, max_q=5,
                    m=1,
                    q=1,
                    seasonal=True,
                    start_P=0,
                    D=None,
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True)
    predictions = model.predict(n_periods=nPeriods)
    return float(predictions.iloc[-1])

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 4000)))