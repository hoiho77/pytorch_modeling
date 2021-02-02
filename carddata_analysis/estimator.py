from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('fpr', fontsize=16)
    plt.ylabel('tpr', fontsize=16)

#roc 그래프 저장
def plot_roc(a, test_labels, test_final_predictions):
    final_fpr, final_tpr, final_thresholds = roc_curve(test_labels, test_final_predictions)
    plot_roc_curve(final_fpr, final_tpr, a)
    plt.legend(loc="lower right", fontsize=16)
    png_name = "D:/user/desktop/python/"+a+"_model.png" #모델명을 활용한 모델 평가 결과 이미지 저장
    plt.savefig(png_name)

#rmse 계산
def estimate_rmse(pred_labels, test_labels):
    md_mse = mean_squared_error(pred_labels, test_labels)
    rmse = np.sqrt(md_mse)
    return rmse

#정확도 계산
def estimate_acc(pred_labels, test_labels):
    acc = accuracy_score(pred_labels, test_labels)
    return acc

#가장 좋은 모델을 비교하여 찾는 함수
def best_md(acc):
    max_values = max(acc.values())
    for key, values in acc.items():
        if values == max_values :
            return key