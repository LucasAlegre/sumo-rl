import numpy as np
import tensorflow as tf

# 1. 准备数据
# 假设我们有输入特征X和对应的交通流量标签Y
X = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
Y = np.array([10, 20, 30])

# 2. 构建神经网络模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 3. 编译模型
model.compile(optimizer='adam', loss='mse')

# 4. 训练模型
model.fit(X, Y, epochs=100, batch_size=1)

# 5. 使用模型进行预测
X_test = np.array([[0.8, 0.9]])
prediction = model.predict(X_test)
print("预测的交通流量：", prediction)