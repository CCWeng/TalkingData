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
scorer = 'roc_auc'



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

n_estimators = 107

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

param_test1b = {
 'max_depth':[4, 6, 7, 8]
 # 'min_child_weight':[1]
}


param_test1c = {
 'max_depth':[4, 9],
 'min_child_weight':[1, 3]
}

gsearch1 = GridSearchCV(
    estimator = XGBClassifier( 
        learning_rate =0.1, 
        n_estimators=n_estimators, 
        max_depth=5,
        min_child_weight=1, 
        gamma=0, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
    param_grid = param_test1b, 
    scoring=scorer,
    # n_jobs=4,
    iid=False, 
    cv=5 )


start_time = time.time()
gsearch1.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

# cv_res1 = cross_val_score(xgb1, trn[use_columns], trn[target], cv=5, scoring='roc_auc')
# cv_res2 = cross_val_score(xgb2, trn[use_columns], trn[target], cv=5, scoring='roc_auc')



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
    scoring=scorer,
    # n_jobs=4,
    iid=False, 
    cv=5)


start_time = time.time()
gsearch2.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time

gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


max_depth = 7
min_child_weight = 1


param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}

gsearch3 = GridSearchCV(
    estimator = XGBClassifier( 
        learning_rate =0.1, 
        n_estimators=n_estimators, 
        max_depth=max_depth,
        min_child_weight=min_child_weight, 
        gamma=0, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1,
        seed=27), 
    param_grid = param_test3, 
    scoring=scorer,
    # n_jobs=4,
    iid=False, 
    cv=5)


start_time = time.time()
gsearch3.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time


gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

gamma=0.0


xgb2 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=max_depth,
    min_child_weight=min_child_weight,
    gamma=gamma,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)

# ml.xgbfit(xgb2, trn, tst, use_columns, printFeatureImportance=False)
ml.xgbfit(xgb2, trn, tst, use_columns, printFeatureImportance=False, early_stopping_rounds=200, target=target)




param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

gsearch4 = GridSearchCV(
    estimator = XGBClassifier( 
        learning_rate =0.1, 
        n_estimators=n_estimators, 
        max_depth=max_depth,
        min_child_weight=min_child_weight, 
        gamma=gamma, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1,
        seed=27), 
    param_grid = param_test4, 
    scoring=scorer,
    # n_jobs=4,
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
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_child_weight=min_child_weight, 
        gamma=gamma, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1,
        seed=27), 
    param_grid = param_test5, 
    scoring=scorer,
    # n_jobs=4,
    iid=False, 
    cv=5)


gsearch5.fit(X_trn, y_trn)

gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

subsample = 0.8
colsample_bytree = 0.8


param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0, 0.1, 1, 100]
 # 'reg_alpha':[0, 1e-6, 1e-5, 1e-4, 1e-3]
 # 'reg_alpha':[0.0005, 0.001, 0.005]
}


param_test6b = {
 'reg_alpha':[0.01, 0.05, 0.1, 0.5]
}

gsearch6 = GridSearchCV(
    estimator = XGBClassifier( 
        learning_rate =0.1, 
        n_estimators=n_estimators, 
        max_depth=max_depth,
        min_child_weight=min_child_weight, 
        gamma=gamma, 
        subsample=subsample, 
        colsample_bytree=colsample_bytree,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1,
        seed=27), 
    param_grid = param_test6b, 
    scoring=scorer,
    # n_jobs=4,
    iid=False, 
    cv=5)

gsearch6.fit(X_trn, y_trn)

gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


reg_alpha = 0.05


xgb3 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=max_depth,
    min_child_weight=min_child_weight,
    gamma=gamma,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    reg_alpha=reg_alpha,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27 )


ml.xgbfit(xgb3, trn, tst, use_columns, printFeatureImportance=False, early_stopping_rounds=200, target=target)



cv_res1 = cross_val_score(xgb1, trn[use_columns], trn[target], cv=5, scoring='roc_auc')
cv_res2 = cross_val_score(xgb2, trn[use_columns], trn[target], cv=5, scoring='roc_auc')

print cv_res1.mean(), cv_res1.std()
print cv_res2.mean(), cv_res2.std()


xgb1.fit(trn[use_columns], trn[target], eval_metric='auc')
xgb3.fit(trn[use_columns], trn[target], eval_metric='auc')
        
p_trn1 = 1 - xgb1.predict_proba(trn[use_columns])[:, 0]
p_tst1 = 1 - xgb1.predict_proba(tst[use_columns])[:, 0]

p_trn2 = 1 - xgb3.predict_proba(trn[use_columns])[:, 0]
p_tst2 = 1 - xgb3.predict_proba(tst[use_columns])[:, 0]

auc_trn1 = roc_auc_score(trn[target], p_trn1)
auc_tst1 = roc_auc_score(tst[target], p_tst1)

auc_trn2 = roc_auc_score(trn[target], p_trn2)
auc_tst2 = roc_auc_score(tst[target], p_tst2)

print auc_trn1, auc_tst1
print auc_trn2, auc_tst2





xgb4 = XGBClassifier(
    learning_rate =0.01,
    n_estimators=5000,
    max_depth=max_depth,
    min_child_weight=min_child_weight,
    gamma=gamma,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    reg_alpha=reg_alpha,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27 )



ml.xgbfit(xgb4, trn, tst, use_columns, printFeatureImportance=False, target=target)



params4 = xgb4.get_params()
params4['eval_metric'] = 'auc' # 0.954
params4['learning_rate'] = 0.1 # 0.952
params4['grow_policy'] = 'lossguide' 
params4['max_leaves'] = 1400
params4['alpha'] = 4
params4['scale_pos_weight'] = 9


params5 = {'learning_rate': 0.3,
          'tree_method': "auto",
          'grow_policy': "lossguide",
          'max_leaves': 1400,  
          'max_depth': 4, 
          'min_child_weight':1,
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'alpha':4,
          'objective': 'binary:logistic', 
          'scale_pos_weight':9,
          'eval_metric': 'auc', 
          'nthread':8,
          'random_state': 99, 
          'silent': True,
          'eval_metric':'auc' }


# xgb5 = XGBClassifier(params)
# ml.xgbfit(xgb5, trn, tst, use_columns, printFeatureImportance=False, target=target, useTrainCV=False)

dtrain = xgb.DMatrix(X_trn, y_trn)
dvalid = xgb.DMatrix(X_tst, y_tst)

gc.collect()

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
model = xgb.train(params5, dtrain, 5000, watchlist, maximize=True, early_stopping_rounds = 500, verbose_eval=1)
pp = model.predict(dvalid, ntree_limit=model.best_ntree_limit)













