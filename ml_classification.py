from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import GridSearchCV
from yellowbrick.classifier import ROCAUC
from sklearn.svm import SVC
import xgboost as xgb
import main


def svm_classifier(X_train, X_test, y_train, y_test):
    svm_model = SVC(C=0.1, kernel='linear', gamma=1, random_state=0)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    main.correct_prediction_number(y_pred, y_test)

    svm_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfussion matrix:\n\n", svm_matrix, sep='')
    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))
    print("\nCohen's Kappa Score:", cohen_kappa_score(y_test, y_pred))
    plot_roc_curve(svm_model, X_train, X_test, y_train, y_test)


    metrics = classification_report(y_test, y_pred, zero_division=1, output_dict=True)

    for name, value in metrics.items():
        print(name, ":", value)

    return


def grid_search_svm(X_train, X_test, y_train, y_test):
    parameters_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['linear', 'poly', 'rbf']}

    grid = GridSearchCV(SVC(), parameters_grid, refit=True, verbose=3)

    grid.fit(X_train, y_train)

    print("\nBest parameters for SVM Classifier:\n", grid.best_params_)

    grid_y_pred = grid.predict(X_test)

    print("\nSuccess metrics of SVM using grid search's result:\n\n", classification_report(y_test, grid_y_pred,
                                                                                            zero_division=1))

    return


def xgboost_classifier(X_train, X_test, y_train, y_test):
    xgb_model = xgb.XGBClassifier(objective='multi:softprob', learning_rate=0.01, max_depth=9, n_estimators=250,
                                  gamma=0.0, min_child_weight=5, use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    main.correct_prediction_number(y_pred, y_test)

    y_pred = [round(value) for value in y_pred]
    xgboost_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfussion matrix:\n\n", xgboost_matrix, sep='')
    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))
    print("\nCohen's Kappa Score:", cohen_kappa_score(y_test, y_pred))
    plot_roc_curve(xgb_model, X_train, X_test, y_train, y_test)

    metrics = classification_report(y_test, y_pred, zero_division=1, output_dict=True)

    for name, value in metrics.items():
        print(name, ":", value)

    return


def grid_search_xgb(X_train, X_test, y_train, y_test):

    parameters_grid = {'max_depth': range(3, 10, 2), 'min_child_weight': range(1, 6, 2),
                       'gamma': [i/10.0 for i in range(0, 5)]}

    grid = GridSearchCV(xgb.XGBClassifier(objective='multi:softprob', n_estimators=250, use_label_encoder=False,
                                          learning_rate=0.01, eval_metric='mlogloss'),
                        parameters_grid, refit=True, verbose=3)

    grid.fit(X_train, y_train)

    print("\nBest parameters for XGBoost Classifier:\n", grid.best_params_)

    grid_y_pred = grid.predict(X_test)

    print("\nSuccess metrics of XGBoost using grid search's result:\n\n", classification_report(y_test, grid_y_pred,
                                                                                                zero_division=1))

    return


def plot_roc_curve(model, X_train, X_test, y_train, y_test):
    visualizer = ROCAUC(model)

    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show()

    return
