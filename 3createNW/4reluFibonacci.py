import math
import random

def relu(x): return max(0.0, x)
def relu_derivative(x): return 1.0 if x > 0 else 0.0

class FibonacciNN:
    def __init__(self, hidden_size=15):
        self.hidden_size = hidden_size
        # Xavier/He initialization (small random values)
        self.w1 = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.b1 = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.w2 = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.b2 = random.uniform(-0.5, 0.5)

    def forward(self, x):
        z = [self.w1[i] * x + self.b1[i] for i in range(self.hidden_size)]
        h = [relu(zi) for zi in z]
        y = sum(self.w2[i] * h[i] for i in range(self.hidden_size)) + self.b2
        return z, h, y

    def train(self, x, target, lr=0.01):
        z, h, y = self.forward(x)
        error = y - target
        dy = error
        
        # Backprop
        for i in range(self.hidden_size):
            dh_i = dy * self.w2[i] * relu_derivative(z[i])
            # Update weights
            self.w2[i] -= lr * dy * h[i]
            self.w1[i] -= lr * dh_i * x
            self.b1[i] -= lr * dh_i
        self.b2 -= lr * dy
        return error ** 2

# 1. 构造斐波那契数据 (Take first 20 numbers)
def get_fib(n):
    a, b = 0, 1
    for _ in range(n): a, b = b, a + b
    return a

MAX_N = 20
# Training targets: log(Fib(n))
training_data = [(n/MAX_N, math.log(get_fib(n))) for n in range(1, MAX_N + 1)]

# 2. 训练
nn = FibonacciNN(hidden_size=20)
for epoch in range(30000):
    loss = sum(nn.train(x, y) for x, y in training_data)
    if epoch % 5000 == 0: print(f"Epoch {epoch}, Loss: {loss:.6f}")

# 3. 验证效果
print("\nPredicting Fibonacci (ReLU + Log Trick):")
for n in range(1, MAX_N + 5): # Predict slightly beyond training range
    x = n / MAX_N
    _, _, y_pred = nn.forward(x)
    pred_val = math.exp(y_pred)
    true_val = get_fib(n)
    error_pct = abs(pred_val - true_val) / true_val * 100
    print(f"F({n}): Pred={pred_val:.1f}, True={true_val}, Error={error_pct:.2f}%")
