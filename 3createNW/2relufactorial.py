import random 
import math

def log_factorial(n):
  return math.log(math.factorial(n))
  
def relu(x):
  return max(0.0, x)

def relu_derivation(y):
  return 1.0 if y > 0 else 0.0


from binascii import b2a_base64
class ReluFactorialNW:
  def __init__(self, hidden_size=10):
    self.hidden_size = hidden_size
    self.w1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
    self.w2 = [random.uniform(-1, 1) for _ in range(hidden_size)]
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
  
  def train(self, x, target, lr=0.001):
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


if __name__ == "__main__":
  MAX_N = 10
  training_data = [
      (n/MAX_N, math.log(math.factorial(n)))
      for n in range(1, MAX_N + 1)
  ]
  nn = ReluFactorialNW()
  
  for epoch in range(20000):
    total_loss = 0.0
    for x, y in training_data:
      total_loss += nn.train(x, y)

    if epoch % 2000 == 0:
      print(f"epoch: {epoch}, loss: {total_loss}")

  for n in range(1,  MAX_N):
    x = n/MAX_N
    z, h, y_pred = nn.forward(x)
    fact_pred = math.exp(y_pred)
    print(f"{n} = {fact_pred:.1f} (true: {math.factorial(n)})")