import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#데이터 가져오기
def load_data():
    csv_path = "D:/user/desktop/python/card.csv"
    data = pd.read_csv(csv_path)
    return data

#상관관계 분석
def corr(data):
    corr = data.corr(method ='pearson')
    print(corr)
    sn.heatmap(corr) #annot=True)
    plt.savefig('D:/user/desktop/python/corr.png')

#범주형 변수 변환(원핫인코딩)
def dummify(data):
    data=pd.get_dummies(data)
    return pd.DataFrame(data)

#데이터 속성이 모두 숫자로 -1, 1으로 데이터 스케일링 진행
def data_scaling(x_num):
    scaling = MinMaxScaler()
    return scaling.fit_transform(x_num)

def num_cat(x_num, x_cat):
    #더미변수화
    x_cat_1 = dummify(x_cat['EDUCATION'])
    x_cat_2 = dummify(x_cat['MARRIAGE'])
    x_cat = pd.concat([x_cat_1, x_cat_2],axis=1)
    cat_names=['EDUCATION1', 'EDUCATION2', 'EDUCATION3', 'EDUCATION4', 'EDUCATION5', 'EDUCATION6', 'MARRIAGE1', 'MARRIAGE2', 'MARRIAGE3']
    x_cat.columns = cat_names
    #print(x_cat)

    #숫자형 데이터 스케일링
    x_num = pd.DataFrame(data_scaling(x_num))
    num_names = ['LIMIT_BAL','AGE','PAY_6','BILL_AMT6','PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4','PAY_AMT5','PAY_AMT6']
    x_num.columns = num_names
    return x_num,x_cat

#main.py와 연결된 함수
def train_test() :
    data = load_data()
    x_num = data[['LIMIT_BAL','AGE','PAY_6','BILL_AMT6','PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4','PAY_AMT5','PAY_AMT6']]
    x_cat = data[['EDUCATION','MARRIAGE']]
    y = data['default']

    #속성 값 정제(스케일링, 더미변수화)
    x_num, x_cat= num_cat(x_num, x_cat)
    x_data = pd.concat([x_num, x_cat],axis=1)

    x_data.to_csv("D:/user/desktop/python/data.csv")

    #테스트셋, 트레이닝 셋 분류
    x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.3)
    print("훈련데이터셋의 수는",len(x_train)," 테스트데이터셋의 수는",len(x_test))
    return  x_train, x_test, y_train, y_test

train_test()