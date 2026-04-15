import random
import math

def relu(x):
  return max(0.0, x)

def sigmoid(x):
    # Clip x to stay within a range math.exp can handle (approx -700 to 700)
    if x < -700:
        return 0.0
    if x > 700:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))

def relu_derivation(y):
  return 1.0 if y > 0 else 0.0

def sigmoid_derivation(y):
  return y * (1 -y)

class ReluFibonacci2LevelNW:
  def __init__(self, hidden_size):
    self.hidden_size = hidden_size
    self.w1 = [random.uniform(-1, 1) for _ in range(self.hidden_size)]
    self.w2 = [random.uniform(-1, 1) for _ in range(self.hidden_size)]
    self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
    self.b2 = random.uniform(-1, 1)
  
  def forward(self, x):
    z = []
    h = []
    for i in range(self.hidden_size):
      zi = self.w1[i] * x + self.b1[i]
      h.append(relu(zi))
      z.append(zi)
    
    y = sum(self.w2[i] * h[i] for i in range(self.hidden_size)) + self.b2
    return z, h, y

  def train(self, x, target, lr=0.01):
    z, h, y = self.forward(x)
    error = y - target

    dy = error
    dz = []
    for i in range(self.hidden_size):
      grad = dy * self.w2[i] * relu_derivation(z[i])
      dz.append(grad)
    
    for i in range(self.hidden_size):
      self.w2[i] -= lr * dy * h[i]
    self.b2 -= lr * dy

    for i in range(self.hidden_size):
      self.w1[i] -= lr * dz[i] * x
      self.b1[i] -= lr * dz[i]
    
    return error ** 2
      

class SigmoidFibonacci2LevelNW:
  def __init__(self, hidden_size):
    self.hidden_size = hidden_size
    self.w1 = [random.uniform(-1, 1) for _ in range(self.hidden_size)]
    self.w2 = [random.uniform(-1, 1) for _ in range(self.hidden_size)]
    self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
    self.b2 = random.uniform(-1, 1)
  
  def forward(self, x):
    z = []
    h = []
    for i in range(self.hidden_size):
      zi = self.w1[i] * x + self.b1[i]
      h.append(sigmoid(zi))
      z.append(zi)
    
    y = sum(self.w2[i] * h[i] for i in range(self.hidden_size)) + self.b2
    return z, h, y

  def train(self, x, target, lr=0.01):
    z, h, y = self.forward(x)
    error = y - target

    dy = error
    dz = []
    for i in range(self.hidden_size):
      grad = dy * self.w2[i] * sigmoid_derivation(h[i])
      dz.append(grad)
    
    for i in range(self.hidden_size):
      self.w2[i] -= lr * dy * h[i]
    self.b2 -= lr * dy

    for i in range(self.hidden_size):
      self.w1[i] -= lr * dz[i] * x
      self.b1[i] -= lr * dz[i]
    
    return error ** 2
      

class Fibonacci3LevelNW:
  def __init__(self, hidden_size):
    self.hidden_size = hidden_size
    self.w1 = [random.uniform(-1, 1) for _ in range(self.hidden_size)]
    self.w2 = [random.uniform(-1, 1) for _ in range(self.hidden_size)]
    self.w3 = [random.uniform(-1, 1) for _ in range(self.hidden_size)]
    self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
    self.b2 = [random.uniform(-1, 1) for _ in range(self.hidden_size)]
    self.b3 = random.uniform(-1, 1)
  
  def forward(self, x):
    z1 = []
    h1 = []
    z2 = []
    h2 = []
    for i in range(self.hidden_size):
      zi = self.w1[i] * x + self.b1[i]
      h1.append(relu(zi))
      z1.append(zi)
    
    for i in range(self.hidden_size):
      zii = self.w2[i] * h1[i] + self.b2[i]
      h2.append(sigmoid(zii))
      z2.append(zii)
    y = sum(self.w3[i] * h2[i] for i in range(self.hidden_size)) + self.b3
    return z1,z2, h1, h2,y

  def train(self, x, target, lr=0.01):
    zz1,z2, h1, h2,y = self.forward(x)
    error = y - target

    dy = error
    dz2 = []
    for i in range(self.hidden_size):
      grad = dy * self.w3[i] * sigmoid_derivation(h2[i])
      dz2.append(grad)
    
    dz1 = []
    for i in range(self.hidden_size):
      grad = dz2[i] * self.w2[i] * relu_derivation(h1[i])
      dz1.append(grad)

    
    for i in range(self.hidden_size):
      self.w3[i] -= lr * dy * h2[i]
    self.b3 -= lr * dy

    for i in range(self.hidden_size):
      self.w1[i] -= lr * dz1[i] * x
      self.b1[i] -= lr * dz1[i]
      self.w2[i] -= lr * dz2[i] * h1[i]
      self.b2[i] -= lr * dz2[i]
    
    return error ** 2

def get_fib(n):
    a, b = 0, 1
    for _ in range(n): a, b = b, a + b
    return a

if __name__ == "__main__":
    training_data = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    MAX_N = 20
    # Training targets: log(Fib(n))
    training_data = [(n/MAX_N, math.log(get_fib(n))) for n in range(1, MAX_N + 1)]

    X = 987
    n1 = ReluFibonacci2LevelNW(hidden_size=20)
    n2 = SigmoidFibonacci2LevelNW(hidden_size=20)
    n3 = Fibonacci3LevelNW(hidden_size=20)

    for epoch in range(20000):
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_loss3 = 0.0
        for x, y in (training_data):
            total_loss1 += n1.train(x, y)
            total_loss2 += n2.train(x, y)
            total_loss3 += n3.train(x, y)
        if epoch % 2000 == 0:
            print(f"n1,  n2,  n3 loss : {total_loss1}_>{total_loss2}_>{total_loss3}")

    _, _,  y_pred1 = n1.forward(18 / MAX_N)
    _, _,  y_pred2 = n2.forward(18 / MAX_N) 
    _, _, _, _, y_pred3 = n3.forward(18 / MAX_N)
    print(math.exp(y_pred1), math.exp(y_pred2), math.exp(y_pred3), get_fib(18))
    print(f"prediction n1, n2, n3 => {y_pred1:.2f}_{y_pred2:.2f}_{y_pred3:.3f}") 