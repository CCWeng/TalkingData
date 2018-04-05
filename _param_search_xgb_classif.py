import xgboost as xgb

from xgboost import XGBClassifier

import model_learning as ml

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, roc_auc_score

my_scorer = make_scorer(roc_auc_score, greater_is_better=True)


n_estimators = 200

clf = XGBClassifier(n_estimators=n_estimators,
                    max_depth=4,
                    objective="binary:logistic",
                    learning_rate=.1, 
                    subsample=.8, 
                    colsample_bytree=.8,
                    gamma=1,
                    reg_alpha=0,
                    reg_lambda=1,
                    nthread=2)

X_trn = trn[use_columns]
y_trn = trn['target']
X_tst = tst[use_columns]
y_tst = tst['target']

clf.fit(X_trn, y_trn, 
        eval_set=[(X_trn, y_trn), (X_tst, y_tst)],
        eval_metric=ml.gini_xgb,
        early_stopping_rounds=None,
        verbose=True)




'''
==========
Ref - 01
==========
'''



# def my_scorer(estimatror, X, y):
#     probs = estimatror.predict_proba(X)[:, 1]
#     score = ml.gini_normalized(y, probs)
#     return score


from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
import time

# %matplotlib inline
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4


# targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

target = 'is_attributed'




#Choose all predictors except target & IDcols

xgb1 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=4,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)


# xgb1.fit(X_trn, y_trn, eval_set=[(X_trn, y_trn), (X_tst, y_tst)], eval_metric='auc', early_stopping_rounds=200)

# pdb.run("ml.xgbfit(xgb1, df_train2, df_val2, use_columns)")
# pdb.run("ml.xgbfit(xgb1, trn, tst, use_columns[:100], printFeatureImportance=False)")
ml.xgbfit(xgb1, trn, tst, use_columns, printFeatureImportance=False, early_stopping_rounds=200, target=target)
ml.xgbfit(xgb1, trn, tst, use_columns, printFeatureImportance=False, early_stopping_rounds=200, target=target, useTrainCV=False)






X_trn = trn[use_columns]
y_trn = trn[target]
X_tst = tst[use_columns]
y_tst = tst[target]


# ===
# Tune max_depth & min_child_weight
# ===

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

# param_test1 = {
#  'max_depth':[2, 3, 4],
#  'min_child_weight':[2, 3, 4]
# }


gsearch1 = GridSearchCV(
    estimator = XGBClassifier( 
        learning_rate =0.1, 
        n_estimators=n_tree, 
        max_depth=5,
        min_child_weight=1, 
        gamma=0, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
    param_grid = param_test1, 
    scoring=make_scorer(roc_auc_score),
    # n_jobs=4,
    iid=False, 
    cv=5 )


start_time = time.time()
gsearch1.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_




param_test2 = {
 'max_depth':[3, 4],
 'min_child_weight':[1, 2, 3]
}

gsearch2 = GridSearchCV(
    estimator = XGBClassifier( 
        learning_rate=0.1, 
        n_estimators=140, 
        max_depth=3,
        min_child_weight=1, 
        gamma=0, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1,
        seed=27), 
    param_grid = param_test2, 
    scoring=ml.my_scorer,
    n_jobs=4,
    iid=False, 
    cv=5)


start_time = time.time()
gsearch2.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time

gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


max_depth = 3
min_child_weight = 1


param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}

gsearch3 = GridSearchCV(
    estimator = XGBClassifier( 
        learning_rate =0.1, 
        n_estimators=140, 
        max_depth=3,
        min_child_weight=1, 
        gamma=0, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1,
        seed=27), 
    param_grid = param_test3, 
    scoring=ml.my_scorer,
    n_jobs=4,
    iid=False, 
    cv=5)


start_time = time.time()
gsearch3.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time


gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

gamma=0.2


xgb2 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=3,
    min_child_weight=1,
    gamma=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)

ml.xgbfit(xgb2, trn, tst, use_columns, printFeatureImportance=False)




param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

gsearch4 = GridSearchCV(
    estimator = XGBClassifier( 
        learning_rate =0.1, 
        n_estimators=59, 
        max_depth=3,
        min_child_weight=1, 
        gamma=0.2, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1,
        seed=27), 
    param_grid = param_test4, 
    scoring=ml.my_scorer,
    n_jobs=4,
    iid=False, 
    cv=5)

gsearch4.fit(X_trn, y_trn)

gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_



param_test5 = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}

gsearch5 = GridSearchCV(
    estimator = XGBClassifier( 
        learning_rate =0.1, 
        n_estimators=59, 
        max_depth=3, 
        min_child_weight=1, 
        gamma=0.2, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1,
        seed=27), 
    param_grid = param_test5, 
    scoring=ml.my_scorer,
    n_jobs=4,
    iid=False, 
    cv=5)


gsearch5.fit(X_trn, y_trn)

gsearch5.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

subsample = 0.8
colsample_bytree = 0.8


param_test6 = {
 # 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
 # 'reg_alpha':[0, 1e-6, 1e-5, 1e-4, 1e-3]
 'reg_alpha':[0.0005, 0.001, 0.005]
}

gsearch6 = GridSearchCV(
    estimator = XGBClassifier( 
        learning_rate =0.1, 
        n_estimators=59, 
        max_depth=3,
        min_child_weight=1, 
        gamma=0.2, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1,
        seed=27), 
    param_grid = param_test6, 
    scoring=ml.my_scorer,
    n_jobs=4,
    iid=False, 
    cv=5)

gsearch6.fit(X_trn, y_trn)

gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


reg_alpha = 0.001


xgb3 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=3,
    min_child_weight=1,
    gamma=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.001,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27 )


ml.xgbfit(xgb3, trn, tst, use_columns, printFeatureImportance=False, early_stopping_rounds=100)





xgb4 = XGBClassifier(
    learning_rate =0.01,
    n_estimators=5000,
    max_depth=3,
    min_child_weight=1,
    gamma=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.001,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27 )



ml.xgbfit(xgb4, trn, tst, use_columns, printFeatureImportance=False)






