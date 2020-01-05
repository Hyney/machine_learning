# 软间隔：对比不同C值带来的差异

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

data, target = datasets.load_iris(return_X_y=True)
X = data[:, [2, 3]]
y = (target == 2).astype(np.float64)
estimator_1 = LinearSVC(C=1, random_state=42)
estimator_2 = LinearSVC(C=100, random_state=42)
scaler = StandardScaler()
svm_estimator_1 = Pipeline(
    [('std', scaler), ('linear_svc', estimator_1)]
)

svm_estimator_2 = Pipeline(
    [('std', scaler), ('linear_svc', estimator_2)]
)

svm_estimator_1.fit(X, y)
svm_estimator_2.fit(X, y)