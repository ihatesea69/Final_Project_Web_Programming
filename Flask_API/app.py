import pandas as pd
import joblib
from tensorflow.keras.models import load_model # type: ignore
from flask import Flask, request, jsonify

# Load mô hình, scaler, và tên cột
model = load_model('./sleep_efficiency_model.h5')
scaler = joblib.load('./scaler.pkl')
columns = joblib.load('./columns.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận dữ liệu từ yêu cầu POST
    data = request.get_json()

    # Chuyển đổi dữ liệu sang DataFrame và sắp xếp theo đúng thứ tự cột
    df = pd.DataFrame(data, index=[0])
    df = df.reindex(columns=columns, fill_value=0)  # Điền 0 nếu thiếu cột

    # Chuẩn hóa dữ liệu
    scaled_data = scaler.transform(df)

    # Dự đoán với mô hình
    prediction = model.predict(scaled_data)

    # Trả về kết quả dự đoán, chuyển float32 sang float
    return jsonify({'sleep_efficiency': float(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True)
