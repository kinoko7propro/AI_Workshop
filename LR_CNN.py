# 基于神经网络下直线模型的线性回归模型代码实现
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 生成足够多的数据点用于训练和拟合
num_points = 1200

data = []
m = 0.2
c = 0.5
for i in range (num_points):
    x = np.random.normal (0.0, 0.8)
    noise = np.random.normal(0.0, 0.04)
    y = m * x + c + noise # 经典的直线模型，加入了部分噪声避免过拟合现象

    data.append([x, y]) # 计算过程

# 经历神经网络迭代后的输入层和输出层需要被分类
x_data = [d[0] for d in data]
y_data = [d[1] for d in data]

x_data = np.array(x_data)
y_data = np.array(y_data)

plt.plot(x_data, y_data, 'ro')
plt.title('Input Data')
plt.show() # 绘制出了图表

# Tensorflow中的Kera模型
model = tf.keras.models.Sequential([ # 采用Sequential模型
    tf.keras.layers.Input(shape = (1, )),
    tf.keras.layers.Dense(1) # 大小为1的全连接层
])

# 神经网路的迭代，迭代过程中模型会计算预测值和真实值的损失，根据优化器调整模型参数
model.compile(optimizer='sgd', loss='mean_squared_error') # 随机梯度下降，这种方法减少了函数的损失，避免陷入局部最优点，同时使用均方误差，对于连续值效果好
num_iterations = 10
for step in range(num_iterations):
    # 开始训练
    model.fit(x_data.reshape(-1, 1), y_data, verbose=0)

    print('\nITERATION', step+1)
    print('W =', model.layers[0].weights[0].numpy()[0][0])
    print('b =', model.layers[0].weights[1].numpy()[0])
    print('loss =', model.loss)

    plt.plot(x_data, y_data, 'ro')

    plt.plot(x_data, model.predict(x_data.reshape(-1, 1)).flatten())

    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.title('Iteration ' + str(step+1) + ' of ' + str(num_iterations))
    plt.show()

