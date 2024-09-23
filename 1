import matplotlib.pyplot as plt

# 数据
years = list(range(2004, 2023))
prices = [
    443.63, 525.62, 602.88, 693.72, 811.05, 972.00, 1121.80, 1383.07, 1700.30,
    2085.42, 2497.27, 2891.16, 3157.70, 3537.96, 3798.45, 4040.00, 4312.00,
    4711.03, 4921.17
]

# 训练模型
def train_model(X, y):
    n = len(X)
    sum_x = sum(X)
    sum_y = sum(y)
    sum_xy = sum(x * y for x, y in zip(X, y))
    sum_x2 = sum(x ** 2 for x in X)
    
    # 使用最小二乘法求解斜率与截距
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept

# 预测函数
def predict(X, slope, intercept):
    return [slope * x + intercept for x in X]

# 画图函数
def plot_data_and_prediction(X, y, predictions):
    plt.scatter(X, y, label='Original Data')
    plt.plot(X, predictions, color='red', label='Prediction Line')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.title('House Price Over Years')
    plt.legend()
    plt.grid(True)
    plt.xticks(X, [str(year) for year in X], rotation=45)
    plt.tight_layout() 
    plt.show()

# 主函数
def main():
    slope, intercept = train_model(years, prices)
    predictions = predict(years, slope, intercept)
    plot_data_and_prediction(years, prices, predictions)

# 运行主函数
main()
