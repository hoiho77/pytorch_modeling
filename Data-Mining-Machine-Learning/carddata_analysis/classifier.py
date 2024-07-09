def train(code, trainset, train_labels, testset):
    # 학습 진행
    if code == 'rf':
        print("랜덤포레스트 모델 훈련 및 데이터 예측 실행")
        forest = RandomForestClassifier(n_estimators=100)
        forest.fit(trainset, train_labels)
        pred_label = forest.predict(testset)
        return pred_label

    elif code == 'lg':
        print("로지스틱 회귀 모델 훈련 및 데이터 예측 실행")
        log_reg = LogisticRegression()
        log_reg.fit(trainset, train_labels)
        pred_label = log_reg.predict(testset)
        return pred_label

    elif code == 'svm':
        print("svm 모델 훈련 및 데이터 예측 실행")
        svm_md = svm.SVC(kernel='linear')
        svm_md.fit(trainset, train_labels)
        pred_label = svm_md.predict(testset)
        return pred_label

    elif code == 'nn':
        print("퍼셉트론 신경망 모델 훈련 및 데이터 예측 실행")
        ppn = Perceptron(max_iter=1000, eta0=0.01, random_state=0)
        ppn.fit(trainset, train_labels)
        pred_label = ppn.predict(testset)
        return pred_label

    elif code == 'dt':
        print("의사결정 나무 모델 훈련 및 데이터 예측 실행")
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(trainset, train_labels)
        pred_label = np.round(tree_reg.predict(testset))  # 분류 알고리즘이기 때문에 0.5를 기준으로 라벨 분류
        return pred_label

    elif code == 'bt':
        print("아다부스트 모델 훈련 및 데이터 예측 실행")
        ada_clf = AdaBoostClassifier(n_estimators=150, algorithm="SAMME.R", learning_rate=0.5)
        ada_clf.fit(trainset, train_labels)
        pred_label = np.round(ada_clf.predict(testset))  # 분류 알고리즘이기 때문에 0.5를 기준으로 라벨 분류
        return pred_label

    else:
        print("해당하는 모델이 없습니다.")
