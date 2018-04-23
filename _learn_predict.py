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
	learning_rate =0.1,
	n_estimators=124,
	max_depth=7,
	min_child_weight=1,
	gamma=0.0,
	subsample=0.8,
	colsample_bytree=0.8,
	reg_alpha=0.05,
	objective= 'binary:logistic',
	nthread=12,
	scale_pos_weight=1,
	seed=27 )


learn_predict_output(xgb4, trn, tst, use_columns, target, 'xgb_180412_01.csv')

# xgb4.fit(trn[use_columns], trn[target])
# pp = xgb4.predict_proba(tst[use_columns])[:, 1]

# tst['is_attributed'] = pp

# d_out = '__output/'
# output_columns = ['click_id', 'is_attributed']

# tst.to_csv(d_output+'xgb_180412_01.csv', columns = output_columns, index=False)




params4 = xgb4.get_params()
params4['eval_metric'] = 'auc' # 0.954

del dtest
gc.collect()
dtrain = xgb.DMatrix(trn[use_columns], trn[target])

watchlist = [(dtrain, 'train')]
xgb4 = xgb.train(params4, dtrain, 200, watchlist, maximize=True, verbose_eval=1)

del dtrain
gc.collect()
dtest = xgb.DMatrix(tst[use_columns])
pp3 = xgb4.predict(dtest, ntree_limit=xgb4.best_ntree_limit)

tst['is_attributed'] = pp3
tst.to_csv('__output/xgb_180412_03.csv', columns = ['click_id', 'is_attributed'], index=False)


del dtest
gc.collect()
x1, x2, y1, y2 = train_test_split(trn[use_columns], trn[target], test_size=0.1, random_state=99)
dtrain = xgb.DMatrix(x1, y1)
dvalid = xgb.DMatrix(x2, y2)
del x1, y1, x2, y2 
gc.collect()
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
# xgb44 = xgb.train(params4, dtrain, 3000, watchlist, maximize=True, early_stopping_rounds = 200, verbose_eval=1)
# xgb445 = xgb.train(params4, dtrain, 3000, watchlist, maximize=True, early_stopping_rounds = 200, verbose_eval=1)
xgb555 = xgb.train(params5, dtrain, 3000, watchlist, maximize=True, early_stopping_rounds = 200, verbose_eval=1)

del dtrain, dvalid
gc.collect()

dtest = xgb.DMatrix(tst[use_columns])
pp = xgb445.predict(dtest, ntree_limit=xgb445.best_ntree_limit)
# pp = model.predict( ntree_limit=20)
tst['is_attributed'] = pp
tst.to_csv('__output/xgb_180415_01.csv', columns = ['click_id', 'is_attributed'], index=False)


del dtest
gc.collect()

dtrain = xgb.DMatrix(trn[use_columns], trn[target])
watchlist = [(dtrain, 'train')]
xgb444 = xgb.train(params4, dtrain, 1000, watchlist, maximize=True, early_stopping_rounds = 200, verbose_eval=1)


## ------------------------

params5 = {
	'learning_rate': 0.3,
	'n_estimators': 28,
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


# xgb5 = XGBClassifier(params5)
# learn_predict_output(xgb5, trn, tst, use_columns, target, 'xgb_180412_02.csv')

dtrain = xgb.DMatrix(trn[use_columns], trn[target])

watchlist = [(dtrain, 'train')]
xgb5 = xgb.train(params5, dtrain, 100, watchlist, maximize=True, verbose_eval=1)

dtest = xgb.DMatrix(tst[use_columns])
# pp = xgb5.predict(dtest, ntree_limit=20)
pp2 = xgb5.predict(dtest, ntree_limit=xgb5.best_ntree_limit)
# pp = model.predict( ntree_limit=20)
tst['is_attributed'] = pp2
tst.to_csv('__output/xgb_180412_02.csv', columns = ['click_id', 'is_attributed'], index=False)



def learn_predict_output(model, trn, tst, use_columns, target, out_file):
	print "Train ..."
	model.fit(trn[use_columns], trn[target])
	print "Predict"
	pp = model.predict_proba(tst[use_columns])[:, 1]
	tst['is_attributed'] = pp
	print "Output"
	out_dir = '__output/'
	output_columns = ['click_id', 'is_attributed']
	tst.to_csv(out_dir+out_file, columns = output_columns, index=False)


pp = xgb4.predict_proba(tst[use_columns])[:, 1]

from sklearn.metrics import roc_auc_score
pp_trn = xgb4.predict_proba(trn[use_columns])[:, 1]
auc_trn = roc_auc_score(trn[target], pp_trn)








