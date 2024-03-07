import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 加载数据
data = pd.read_csv('data.csv')

# 将日期和时间列合并为一个时间戳
data['Timestamp'] = pd.to_datetime(data['Date'], format='%Y%m%d%H%M')

# 对经纬度进行归一化处理
scaler = MinMaxScaler()
data[['Lat', 'Lon']] = scaler.fit_transform(data[['Lat', 'Lon']])

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data[['Lat', 'Lon', 'Pres']], data['Wind'], test_size=0.2, random_state=42)

# 训练XGBoost模型
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01, random_state=42)
xgb.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = xgb.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

new_data = pd.DataFrame({'Lat': [24.5], 'Lon': [127.5], 'Pres': [975]})
new_data_pred = xgb.predict(new_data)
print(f'New data prediction: {new_data_pred}')

import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 加载数据
data = pd.read_csv('data.csv')

# 提取特征和目标值
features = data[['Lat', 'Lon', 'Pres']].values
target = data[['Wind']].values

# 数据归一化
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_target = scaler.fit_transform(target)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(scaled_features, scaled_target,
                                                    test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.Tensor(X_train).unsqueeze(1)
y_train_tensor = torch.Tensor(y_train).unsqueeze(1)
X_test_tensor = torch.Tensor(X_test).unsqueeze(1)
y_test_tensor = torch.Tensor(y_test).unsqueeze(1)


# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 定义模型参数
input_size = 3 # 输入特征维度
hidden_size = 32  # 隐藏层维度
num_layers = 2  # LSTM层数
output_size = 1  # 输出维度

# 实例化模型
model = LSTM(input_size, hidden_size, num_layers, output_size)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor.to(device))
    loss = criterion(outputs, y_train_tensor.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}')


# 使用模型进行预测
new_data = [[24.5, 127.5, 975]]  # 待预测的新数据

model.eval()
with torch.no_grad():
    prediction = model(torch.Tensor(new_data).unsqueeze(0).to(device))
    unscaled_prediction = scaler.inverse_transform(prediction.cpu().numpy())
    print(f'Wind Speed Prediction for New Data: {unscaled_prediction[0][0]:.2f}')