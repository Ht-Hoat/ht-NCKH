import numpy as np
import matplotlib.pyplot as plt

# 1. Định nghĩa hàm loss
# Giả sử mình có hàm loss L(θ) = θ⁴ - 6θ² + 4θ + 20
# khai báo hàm loss
def cal_loss_function_theta(theta):
    return theta**4 - 6*theta**2 + 4*theta + 20

# Đạo hàm của hàm Loss L'(θ) = 4θ³ - 12θ + 4 (đạo hàm của θ⁴ - 6θ² + 4θ + 20))
def cal_loss_derivative_theta(theta):
    return 4*theta**3 - 12*theta + 4

# 2. Vẽ đồ thị hàm loss
# khai báo khoảng giá trị của theta
theta_values = np.linspace(-3, 3, 400)

# giá trị của hàm loss theo khoảng theta
loss_values = cal_loss_function_theta(theta_values)

# Vẽ đồ thị loss
plt.plot(theta_values, loss_values)

# 3. Khởi tạo
# Khởi tạo giá trị ban đầu của theta
theta = 2.8

# Khởi tạo siêu tham số - hyperparameters
learning_rate = 0.02

N = 100 # số vòng lặp

# khai báo thêm (for momentum)
beta = 0.9
v = 0 # Initialize velocity vector . Biến v được khởi tạo là 0, đây chính là vận tốc tích lũy của theta.

# cách 2: Khai báo số vòng lặp
for i in range(N):
    # tính giá trị hàm loss
    loss = cal_loss_function_theta(theta)

    # tính giá trị đạo hàm của hàm loss theo theta
    grad = cal_loss_derivative_theta(theta)

    # Vẽ điểm theta hiện tại
    plt.plot(theta, loss, 'ro') # ro là điểm tròn màu đỏ

    # Dừng để quan sát
    plt.pause(0.1)

    # Tính vận tốc
    v = beta * v + (1 - beta) * grad

    # Cập nhật tham số theta dựa vào vận tốc
    theta = theta - learning_rate * v

    print("Giá trị theta tối ưu::::", theta)

plt.show()