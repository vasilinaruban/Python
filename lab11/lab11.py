import torch
import torch.nn.functional as F

class Layer:
    def __init__(self, size1, size2):
        self.w = torch.randn(size1, size2, requires_grad=True)
        self.b = torch.randn(size2, requires_grad=True)

class CustomBareboneModel:
    def __init__(self):
        self.l1 = Layer(256, 64)
        self.l2 = Layer(64, 16)
        self.l3 = Layer(16, 4)

    def forward(self, x):
        x = F.relu(torch.matmul(x, self.l1.w) + self.l1.b)
        x = torch.tanh(torch.matmul(x, self.l2.w) + self.l2.b)
        x = F.softmax(torch.matmul(x, self.l3.w) + self.l3.b, dim=1)
        return x

    def backward(self, x, target):
        output = self.forward(x)
        loss = F.cross_entropy(output, target)

        loss.backward()

        lr = 0.01
        self.l1.w.data -= lr * self.l1.w.grad
        self.l1.b.data -= lr * self.l1.b.grad
        self.l2.w.data -= lr * self.l2.w.grad
        self.l2.b.data -= lr * self.l2.b.grad
        self.l3.w.data -= lr * self.l3.w.grad
        self.l3.b.data -= lr * self.l3.b.grad

        self.l1.w.grad.zero_()
        self.l1.b.grad.zero_()
        self.l2.w.grad.zero_()
        self.l2.b.grad.zero_()
        self.l3.w.grad.zero_()
        self.l3.b.grad.zero_()


x = torch.randn(1, 256)
target = torch.tensor([0], dtype=torch.long)
model = CustomBareboneModel()
print("Before backward pass:")
print(model.forward(x))

model.backward(x, target)

print("\nAfter backward pass:")
print(model.forward(x))