import math
import random

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)


class FactorialNW:
    def __init__(self, hidden_size=10):
        self.hidden_size = hidden_size
        self.w1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.w2 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b2 = random.uniform(-1, 1)
    
    def forward(self, x):
        h = []
        for i in range(self.hidden_size):
            z = self.w1[i] * x + self.b1[i]
            h.append(sigmoid(z))
        
        y = sum(self.w2[i] * h[i] for i in range(self.hidden_size)) + self.b2
        return h, y

    def train(self, x,  target, lr=0.01):
        h, y = self.forward(x)
        error = y - target

        dy = error
        dh = [
            dy * self.w2[i] * sigmoid_derivative(h[i])
            for i in range(self.hidden_size)
        ]

        for i in range(self.hidden_size):
            self.w2[i] -= lr * dy * h[i]
        self.b2 -= lr * dy

        for i in range(self.hidden_size):
            self.w1[i] -= lr * dh[i] * x
            self.b1[i] -= lr * dh[i]
        
        return error ** 2


MAX_N = 10
t = random.uniform(-1,1)
print(t)

def log_factorial(n):
    return math.log(math.factorial(n))

training_data = [
    (n / MAX_N, log_factorial(n))
    for n in range(1, MAX_N + 1)
]
print("t", training_data)

nn = FactorialNW(hidden_size=10)

for epoch in range(20000):
    loss = 0
    for x, y in training_data:
        loss += nn.train(x, y)
    
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nPrediction")
for n in range(1,  MAX_N + 1):
    x = n / MAX_N
    _, y_pred = nn.forward(x)
    
    fac_pred = math.exp(y_pred)
    print(f"{n} = {fac_pred:.1f} (true: {math.factorial(n)})")