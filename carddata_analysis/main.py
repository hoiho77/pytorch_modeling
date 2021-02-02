from dataset import train_test  # 데이터셋과 관련된 모듈
from classifier import *  # 머신러닝 모델과 관련된 모듈
from estimator import *

if __name__ == '__main__':
    acc, rmse = {}, {}
    x_train, x_test, y_train, y_test = train_test()
    algorithms = input("\n활용하고자 하는 알고리즘(lg, rf, dt, nn, svm, bt)을 입력하시오. \n\n복수 입력 시 띄어쓰기를 해주세요 :").split()
    print("")

    for md_name in algorithms:
        pred_labels = train(md_name, x_train, y_train, x_test)
        rmse[md_name] = estimate_rmse(pred_labels, y_test)
        acc[md_name] = estimate_acc(pred_labels, y_test)

        # 모델별 roc 커브 생성
        plot_roc(md_name, pred_labels, y_test)

    print("훈련된 모델의 roc커브의 이미지가 저장되었습니다.\n")

    for md_name in algorithms:
        print(md_name, "알고리즘을 활용하여 훈련한 모델의 rmse: %.3f" % rmse[md_name], "acc: %.2f" % (acc[md_name] * 100), "%")

    print("\n*** 가장 좋은 모델은", best_md(acc), ", 정확도는 %.2f" % (max(acc.values()) * 100), "% ***")

