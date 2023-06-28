import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

x_train.shape

# requires_grad=True 학습을 통해 값이 변경되는 변수
W = torch.zeros(1, requires_grad=True) # 가중치
b = torch.zeros(1, requires_grad=True) # 편향


# 경사하강법 구현하기
optimizer = optim.SGD([W, b], lr = 0.01) # SGD 경사하강법

n_epochs = 1000
for epoch in range(n_epochs+1):
    hypothesis = x_train * W + b

    cost = torch.mean((hypothesis - y_train) ** 2)  # MSE

    # gradient를 0으로 초기화
    optimizer.zero_grad()

    # 비용함수를 미분하여 gradient 계산
    cost.backward()

    #W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
        print(f'epoch {epoch}/{n_epochs} W :{W.item()}, b :{b.item()}, Cost{cost.item()}')

# 자동미분 실습하기
w = torch.tensor(2.0, requires_grad=True)
y = w**2
z = 2*y+5

z.backward()
print(f'수식을 w로 미분한 값 {w.grad}')
