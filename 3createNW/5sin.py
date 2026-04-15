import math
import random

# 使用 tanh 激活函数，因为它在 [-1, 1] 之间表现比 ReLU 更平滑，适合正弦波
def tanh(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1.0 - math.tanh(x)**2

class SineNN:
    def __init__(self, hidden_size=10):
        self.hidden_size = hidden_size
        # 初始化权重：由于 sin 值域小，初始化在 [-0.5, 0.5] 即可
        self.w1 = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.b1 = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.w2 = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.b2 = random.uniform(-0.5, 0.5)

    def forward(self, x):
        # 隐藏层
        z = [self.w1[i] * x + self.b1[i] for i in range(self.hidden_size)]
        h = [tanh(zi) for zi in z]
        # 输出层（线性输出）
        y = sum(self.w2[i] * h[i] for i in range(self.hidden_size)) + self.b2
        return z, h, y

    def train(self, x, target, lr=0.01):
        z, h, y = self.forward(x)
        error = y - target
        
        # 反向传播更新
        for i in range(self.hidden_size):
            # 计算隐藏层梯度
            grad_h = error * self.w2[i] * tanh_derivative(z[i])
            # 更新输出权重
            self.w2[i] -= lr * error * h[i]
            # 更新输入权重和偏置
            self.w1[i] -= lr * grad_h * x
            self.b1[i] -= lr * grad_h
        self.b2 -= lr * error
        return error ** 2

# 构造训练数据：在 [0, 2π] 之间随机采样
train_data = []
for _ in range(500):
    x_val = random.uniform(0, 2 * math.pi)
    train_data.append((x_val, math.sin(x_val)))

# 训练网络
nn = SineNN(hidden_size=15)
for epoch in range(10000):
    total_loss = sum(nn.train(x, y) for x, y in train_data)
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Avg Loss: {total_loss/len(train_data):.6f}")

# 测试效果
print("\nSine Wave Prediction Results:")
test_points = [0, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]
for x in test_points:
    _, _, y_pred = nn.forward(x)
    print(f"sin({x:.2f}): Pred={y_pred:.4f}, True={math.sin(x):.4f}")
