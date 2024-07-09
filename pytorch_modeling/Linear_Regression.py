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


# nn.Module로 선형회귀 구현

import torch.nn as nn
#model = nn.Linear(input_dim, output_dim)
model = nn.Linear(1, 1)

print(list(model.parameters()))

import torch.nn.funtional as F
cost = F.mse_loss(prediction, y_train)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward() # backward 연산
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))

# 모델을 활용하여 예측
new_var = torch.FloatTensor([[4.0]])
pred_y = model(new_var)
print(pred_y)
print(list(model.parameters()))