import matplotlib.pyplot as plt

# 阶乘函数
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

# 多项式函数
def polynomial(x, params):
    return params[0] + params[1]*x + params[2]*x**2 + params[3]*x**3 + params[4]*x**4 + params[5]*x**5
# 损失函数
def loss_function(params, x_values, y_values):
    total_error = 0
    for x, y in zip(x_values, y_values):
        prediction = polynomial(x, params)
        total_error += (prediction - y) ** 2
    return total_error / len(x_values)

# 梯度下降算法更新参数
def gradient_descent(x_values, y_values):
    params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 初始化参数列表
    learning_rate = 0.0001
    iterations = 50000
    for _ in range(iterations):
        gradients = [0.0] * 6  # 初始化梯度列表
        for x, y in zip(x_values, y_values):
            prediction = polynomial(x, params)
            error = prediction - y
            gradients[0] += error * 1
            gradients[1] += error * x
            gradients[2] += error * x**2
            gradients[3] += error * x**3
            gradients[4] += error * x**4
            gradients[5] += error * x**5
        
        # 计算平均梯度并更新参数
        for i in range(6):
            gradients[i] /= len(x_values)
            params[i] -= learning_rate * gradients[i]
    
    return params

# 初始化参数
parameters = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# 生成样本数据
x_values = [i * 0.1 for i in range(-31, 32)]
y_values = [sum([((-1)**k * x**(2*k+1)) / factorial(2*k+1) for k in range(10)]) for x in x_values]

# 运行梯度下降算法优化参数
optimized_parameters = gradient_descent(x_values, y_values)

# 绘制原始样本点和预测函数曲线
plt.figure(figsize=(10, 5))
plt.scatter(x_values, y_values, label='Original sin(x) Samples', color='blue')
predicted_y_values = [polynomial(x, optimized_parameters) for x in x_values]
plt.plot(x_values, predicted_y_values, label='Predicted Function', color='red')
plt.title('Fit of sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
