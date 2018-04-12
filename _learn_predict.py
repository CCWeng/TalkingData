from sklearn.linear_model import LogisiticRegression


model_lr = LogisiticRegression(penalty='l1')

cv_res = cross_val_score(model_lr, trn[use_columns], trn[target], cv=5, scoring='roc_auc')
print 'Results of cross-validation :', cv_res.mean(), cv_res.std()


model_lr.fit(trn[use_columns], trn[target])
pp = model.predict_proba(tst[use_columns])[:, 1]

tst['is_attributed'] = pp

d_out = '__output/'
output_columns = ['click_id', 'is_attributed']

tst.to_csv(d_output+'lr_180403_01.csv', columns = output_columns, index=False)



# == xgboost
import xgboost as xgb

from xgboost import XGBClassifier


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


xgb4.fit(trn[use_columns], trn[target])
pp = xgb4.predict_proba(tst[use_columns])[:, 1]

tst['is_attributed'] = pp

d_out = '__output/'
output_columns = ['click_id', 'is_attributed']

tst.to_csv(d_output+'xgb_180412_01.csv', columns = output_columns, index=False)


