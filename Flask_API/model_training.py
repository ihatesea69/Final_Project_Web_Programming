import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Tải và tiền xử lý dữ liệu
df = pd.read_csv('./Sleep_Efficiency_cleaned.csv')

# Mã hóa các biến phân loại
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_smoking = LabelEncoder()
df['Smoking status'] = le_smoking.fit_transform(df['Smoking status'])

# Định nghĩa X và y
X = df.drop(columns=['Sleep efficiency', 'ID', 'Bedtime', 'Wakeup time'])
y = df['Sleep efficiency']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lưu các tên cột
columns = X.columns
joblib.dump(columns, 'columns.pkl')

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Xây dựng mô hình
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# Biên dịch mô hình
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Lưu mô hình và scaler
model.save('sleep_efficiency_model.h5')
joblib.dump(scaler, 'scaler.pkl')
