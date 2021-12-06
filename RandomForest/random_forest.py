from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


def voting():
    """硬投票"""
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

    log_estimator = LogisticRegression(random_state=42)
    forest_estimator = RandomForestClassifier(random_state=42)
    svm_estimator = SVC(random_state=42)

    # 硬投票实验
    voting = VotingClassifier([('lr', log_estimator), ('rf', forest_estimator), ('svm', svm_estimator)], voting='hard') # 投票分类器：硬投票
    voting.fit(x_train, y_train)
    for clf in (log_estimator, forest_estimator, svm_estimator, voting):
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred), '\n')


def soft_voting():
    """软投票"""
    # 软投票：要求必须各个分类器都能得出概率值
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

    log_estimator = LogisticRegression(random_state=42)
    forest_estimator = RandomForestClassifier(random_state=42)
    svm_estimator = SVC(probability=True, random_state=42)

    voting = VotingClassifier([('lr', log_estimator), ('rf', forest_estimator), ('svm', svm_estimator)], voting='soft') # 投票分类器：软投票
    voting.fit(x_train, y_train)
    for clf in (log_estimator, forest_estimator, svm_estimator, voting):
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred), '\n')


def bagging():
    """Bagging策略实验"""
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

    bagging = BaggingClassifier(
        DecisionTreeClassifier(),
        n_estimators=10,
        max_samples=100,
        bootstrap=True,
        # n_jobs=-1,
        random_state=42
    )
    bagging.fit(x_train, y_train)
    y_pred = bagging.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print(score)

    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(x_train, y_train)
    y_pred = decision_tree.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print(score)
#
# # 决策边界
#     # 集成与传统方法对比
def plot_decision_boundary(classifier, x_data, y_data, axes=(-1.5, 2.5, -1.5, 2.5), alpha=0.5, contour=True):
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

    x1 = np.linspace(axes[0], axes[1], 100)
    x2 = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1, x2)
    x_new = np.c_[x1.ravel(), x2.ravel()]
    y_predict = classifier.predict(x_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_predict, cmap=custom_cmap, alpha=0.3)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contour(x1, x2, y_predict, cmap=custom_cmap2, alpha=0.8)

    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'yo', alpha=0.5)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs', alpha=0.5)
    plt.axis(axes)
    plt.xlabel('x1')
    plt.ylabel('x2')


def decision_bagging_compare():
    """decision_tree with Bagging and decision_tree compare"""
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)
    bagging = BaggingClassifier(
        DecisionTreeClassifier(),
        n_estimators=10,
        max_samples=100,
        bootstrap=True,
        # n_jobs=-1,
        random_state=42
    )
    bagging.fit(x_train, y_train)
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(x_train, y_train)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plot_decision_boundary(decision_tree, X, y)
    plt.title('Decision Tree')

    plt.subplot(122)
    plot_decision_boundary(bagging, X, y)
    plt.title('Decision Tree With Bagging')
    plt.show()


def oob_bagging():
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)
    oob_bagging = BaggingClassifier(
        DecisionTreeClassifier(),
        n_estimators=10,
        max_samples=100,
        bootstrap=True,
        # n_jobs=-1,
        random_state=42,
        oob_score=True
    )
    oob_bagging.fit(x_train, y_train)
    # score = oob_bagging.oob_score_
    y_pred = oob_bagging.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print(score)

# mnist = fetch_mldata('MNIST original')
# # # 随机森林
# # iris_data = load_iris()
# rf_clf = RandomForestClassifier(n_estimators=500)
# # # rf_clf.fit(iris_data.data, iris_data.target)
# # # for name, score in zip(iris_data.feature_names, rf_clf.feature_importances_):
# # #     print(name, score)
# rf_clf.fit(mnist.data, mnist.target)
# # rf_clf.feature_importances_
# def plot_digit(data):
#     image = data.reshape(28, 28)
#     plt.show(image)
#     plt.axis('off')
#
# plot_digit(rf_clf.feature_importances_)
# plt.colorbar(ticks=[rf_clf.feature_importances_.min(), rf_clf.feature_importances_.max()])
# plt.show()

# Boosting-提升策略
# AdaBoost

# Gradient Boosting---梯度提升


def gradient_boost():
    """梯度提升实验展示"""
    np.random.seed(42)
    X = np.random.rand(100, 1) - 0.5
    y = 3 * X ** 2 + 0.05 * np.random.randn(100)
    print(y.shape)
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(X, y)

    y2 = y - tree_reg.predict(X)
    tree_reg2 = DecisionTreeRegressor()
    tree_reg2.fit(X, y2)

    y3 = y2 - tree_reg2.predict(X)
    tree_reg3 = DecisionTreeRegressor()
    tree_reg3.fit(X, y3)

    X_new = np.array([[0.8]])
    y_predict = sum(tree.predict(X_new) for tree in (tree_reg, tree_reg2, tree_reg3))
    print(y_predict)


def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)


def visual(regressors, data, target):
    """GBDT提升对比实验可视化"""
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plot_predictions([regressors[0]], data, target, axes=[-0.5, 0.5, -0.1, 0.8], label='Ensemble predictions')
    plt.title('learning_rate={}, n_estimators={}'.format(regressors[0].learning_rate, regressors[0].n_estimators))
    plt.subplot(122)
    plot_predictions([regressors[1]], data, target, axes=[-0.5, 0.5, -0.1, 0.8], label='Ensemble predictions')
    plt.title('learning_rate={}, n_estimators={}'.format(regressors[1].learning_rate, regressors[1].n_estimators))
    plt.show()


def gradient_boosting():
    """GBDT提升对比实验"""
    np.random.seed(42)
    X = np.random.rand(100, 1) - 0.5
    y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)
    estimator = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
    estimator.fit(X, y)
    estimator_1 = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=0.1, random_state=42)
    estimator_1.fit(X, y)
    visual((estimator, estimator_1), X, y)
    estimator_2 = GradientBoostingRegressor(max_depth=2, n_estimators=100, learning_rate=0.1, random_state=42)
    estimator_2.fit(X, y)
    visual((estimator_1, estimator_2), X, y)


def stop_visual(errors, bst_n_estimators, min_error, gbrt_best, data, target):
    """提前停止策略可视化"""
    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plt.plot(errors, 'b.-')
    plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], 'k--')
    plt.plot([0, 120], [min_error, min_error], 'k--')
    plt.axis([0, 120, 0, 0.01])
    plt.title('Val Error')

    plt.subplot(122)
    plot_predictions([gbrt_best], data, target, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title('Best Model(%d trees)' % bst_n_estimators)
    plt.show()


def stop():
    """提前停止策略"""
    from sklearn.metrics import mean_squared_error
    np.random.seed(42)
    X = np.random.rand(100, 1) - 0.5
    y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)
    estimator = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
    estimator.fit(x_train, y_train)
    errors = [mean_squared_error(y_test, y_predict) for y_predict in estimator.staged_predict(x_test)]
    print(errors)
    bst_n_estimators = np.argmin(errors)
    print(bst_n_estimators)

    estimator_test = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
    estimator_test.fit(x_train, y_train)
    min_error = np.min(errors)
    print(min_error)

    stop_visual(errors, bst_n_estimators, min_error, estimator_test, X, y)


if __name__ == '__main__':
    # gradient_boosting()
    stop()




