import model_learning as ml

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score



target='is_attributed'

model_lr = LogisticRegression()

use_columns = list()
use_columns += oof_columns
use_columns += smooth_columns



res = cross_val_score(model_lr, trn[oof_columns], trn[target], cv=5, scoring='roc_auc')
print res.mean(), ',', res.std()
# 0.930952032404 , 0.0154225260174

res = cross_val_score(model_lr, trn[smooth_columns], trn[target], cv=5, scoring='roc_auc')
print res.mean(), ',', res.std()
## 0.965031377011 , 0.00675576470674

res = cross_val_score(model_lr, trn[use_columns], trn[target], cv=5, scoring='roc_auc')
print res.mean(), ',', res.std()
## 0.957176464157 , 0.00945298845327

res = cross_val_score(model_lr, trn[use_columns], trn[target], cv=5, scoring='roc_auc')
print "all in use : %.3f, %.3f" % (res.mean(), res.std())
del_columns = list()
for del_col in use_columns:
	use_columns2 = list(use_columns)
	use_columns2.remove(del_col)
	res2 = cross_val_score(model_lr, trn[use_columns2], trn[target], cv=5, scoring='roc_auc')
	delta = res2.mean()-res.mean()
	print "remove %s: %.3f, %.3f (delta=%.5f)" % (del_col, res2.mean(), res2.std(), delta)
	if delta > 0:
		del_columns.append(del_col)


use_columns2 = list(set(use_columns).difference(del_columns))
res = cross_val_score(model_lr, trn[use_columns2], trn[target], cv=5, scoring='roc_auc')
print res.mean(), ',', res.std()
## 0.967688982404 , 0.00473385416347

'''
smooth + oof - del > smooth > smooth + oof > oof
'''

use_columns = list()
use_columns += smooth_columns
use_columns += dumm_click_hour_columns

res = cross_val_score(model_lr, trn[use_columns], trn[target], cv=5, scoring='roc_auc')
print res.mean(), ',', res.std()
## 0.837369368582 , 0.0406298519356


t1 = trn.groupby('click_hour')['is_attributed'].mean()
t2 = tst.groupby('click_hour')['is_attributed'].mean()
tmp = pd.concat([t1, t2], axis=1)
tmp.columns = ['train', 'test']

res = cross_val_score(model_lr, trn[smooth_columns], trn[target], cv=5, scoring='roc_auc')
print "smooth : %.3f, %.3f" % (res.mean(), res.std())
for col in dumm_click_hour_columns:
	use_columns2 = smooth_columns + [col]
	res2 = cross_val_score(model_lr, trn[use_columns2], trn[target], cv=5, scoring='roc_auc')
	delta = res2.mean()-res.mean()
	print "add %s: %.3f, %.3f (delta=%.5f)" % (col, res2.mean(), res2.std(), delta)



use_columns = list()
use_columns += smooth_columns
use_columns += smooth_hour_columns
use_columns += oof_hour_columns

res = cross_val_score(model_lr, trn[use_columns], trn[target], cv=5, scoring='roc_auc')
print res.mean(), ',', res.std()
# 0.964951970581 , 0.00674139463969

use_columns = list()
use_columns += smooth_columns
use_columns += smooth_hour_columns

res = cross_val_score(model_lr, trn[use_columns], trn[target], cv=5, scoring='roc_auc')
print res.mean(), ',', res.std()
# 0.964955693786 , 0.00672658895775

use_columns = list()
use_columns += smooth_columns
use_columns += oof_hour_columns

res = cross_val_score(model_lr, trn[use_columns], trn[target], cv=5, scoring='roc_auc')
print res.mean(), ',', res.std()
# 0.964900608062 , 0.00675434866827


## -- 
res = cross_val_score(model_lr, trn[rescaled_smooth_columns], trn[target], cv=5, scoring='roc_auc')
print res.mean(), ',', res.std()
## 0.962129222083 , 0.00653315427716


use_columns = list()
use_columns += rescaled_smooth_columns
use_columns += dumm_click_hour_columns

res = cross_val_score(model_lr, trn[use_columns], trn[target], cv=5, scoring='roc_auc')
print res.mean(), ',', res.std()
## 0.840575077855 , 0.0384845773222


res = cross_val_score(model_lr, trn[dumm_click_hour_columns], trn[target], cv=5, scoring='roc_auc')
print res.mean(), ',', res.std()


model_lr2 = LogisticRegression(C=0.1, solver='sag')
res = cross_val_score(model_lr2, trn[rescaled_smooth_columns], trn[target], cv=5, scoring='roc_auc')
print res.mean(), ',', res.std()
## 0.962129950862 , 0.00653066747872

model_lr3 = LogisticRegression(penalty='l1')
res = cross_val_score(model_lr3, trn[rescaled_smooth_columns], trn[target], cv=5, scoring='roc_auc')
print res.mean(), ',', res.std()



use_columns = list()
use_columns += rescaled_smooth_columns
use_columns += rescaled_dumm_hour_columns

res = cross_val_score(model_lr, trn[use_columns], trn[target], cv=5, scoring='roc_auc')
print res.mean(), ',', res.std()
## 0.835578882563 , 0.0413137578618



for i in range(1, len(stats_columns2)+1):
	add_columns = stats_columns2[:i]
	res = cross_val_score(model_lr, trn[smooth_columns+add_columns], trn[target], cv=5, scoring='roc_auc')
	print i, ':', res.mean(), ',', res.std()


res = cross_val_score(model_lr, trn[smooth_columns], trn[target], cv=5, scoring='roc_auc')
print "smooth : %.3f, %.3f" % (res.mean(), res.std())
for col in stats_columns2:
	use_columns2 = smooth_columns + [col]
	res2 = cross_val_score(model_lr, trn[use_columns2], trn[target], cv=5, scoring='roc_auc')
	delta = res2.mean()-res.mean()
	print "add %s: %.3f, %.3f (delta=%.5f)" % (col, res2.mean(), res2.std(), delta)


tmp_columns = smooth_columns + ['is_attributed_mean_on_ip']
res = cross_val_score(model_lr, trn[tmp_columns], trn[target], cv=5, scoring='roc_auc')
print res.mean(), ',', res.std()
## 0.996944962303 , 0.000308038420449

tmp_columns = smooth_columns + ['oof_is_attributed@ip']
res = cross_val_score(model_lr, trn[tmp_columns], trn[target], cv=5, scoring='roc_auc')
print res.mean(), ',', res.std()
## 0.964786914585 , 0.00683090301055


##-- cate_freq_columns

from sklearn.metrics import roc_auc_score

z_statistics = pd.Series()
for c in cate_freq_columns:
	z = fa.Ztest(trn, 'is_attributed', c)
	z_statistics[c] = z

z_statistics.sort_values(ascending=False, inplace=True)


z_statistics2 = pd.Series()
for c in cate_freq_columns:
	z = fa.Ztest_bi(trn, 'is_attributed', c)
	z_statistics2[c] = z

z_statistics2.sort_values(ascending=False, inplace=True)


add_columns = z_statistics.index.tolist()

res = cross_val_score(model_lr, trn[smooth_columns], trn[target], cv=5, scoring='roc_auc')
print "smooth : %.3f, %.3f" % (res.mean(), res.std())
ml.auc_score(model_lr, trn[smooth_columns], trn[target], tst[smooth_columns], tst[target])
for col in add_columns:
	columns = smooth_columns+[col]
	res2 = cross_val_score(model_lr, trn[columns], trn[target], cv=5, scoring='roc_auc')
	delta = res2.mean()-res.mean()
	print "--\nadd %s: %.3f, %.3f (delta=%.5f)" % (col, res2.mean(), res2.std(), delta)
	ml.auc_score(model_lr, trn[columns], trn[target], tst[columns], tst[target])



col = smooth_columns[0]

for col in ['app_freq']:
# for col in smooth_columns:
	plt.subplot(211)
	trn.groupby('is_attributed')[col].plot('density')
	plt.legend(['is_attirbuted=0', 'is_attirbuted=1'])
	plt.title('train')

	plt.subplot(212)
	tst.groupby('is_attributed')[col].plot('density')
	plt.legend(['is_attirbuted=0', 'is_attirbuted=1'])
	plt.title('test')

	plt.suptitle(col)
	plt.show()



plt.subplot(211)
plt.plot(trn.sort_values(by='app_freq', ascending=False)[target])
plt.title('train')
plt.subplot(212)
plt.plot(tst.sort_values(by='app_freq', ascending=False)[target])
plt.title('test')
plt.show()


for i in [x*10000 for x in range(1, 6)]:
	print '--'
	print i
	ml.auc_score(model_lr, trn[smooth_columns], trn[target], tst.iloc[:i][smooth_columns], tst.iloc[:i][target])


## try SGDClassifier
# model_lr = LogisticRegression()
model_lr = LogisticRegression(class_weight='balanced')

model_sgd = SGDClassifier()

columns = smooth_columns
columns = smooth_columns + ['app_freq']

ml.auc_score(model_lr, trn[columns], trn[target], tst[columns], tst[target])
ml.auc_score(model_sgd, trn[columns], trn[target], tst[columns], tst[target], has_predict_proba=False)


## -- 
from sklearn.metrics import accuracy_score

y = trn[target]

# columns1 = smooth_columns
# columns2 = smooth_columns + ['app_freq']
# columns3 = use_columns
columns1 = oof_columns
columns2 = smooth_columns
columns3 = stats_columns
columns4 = tgt_stats_columns




# model1 = LogisticRegression(class_weight='balanced', fit_intercept=False)
# model2 = LogisticRegression(class_weight='balanced', fit_intercept=False)

# model1 = LogisticRegression(fit_intercept=False)
# model2 = LogisticRegression(fit_intercept=False)

# model1 = LogisticRegression(solver='newton-cg', max_iter=100000)
# model2 = LogisticRegression(solver='newton-cg', max_iter=100000)

model1 = LogisticRegression(penalty='l1')
model2 = LogisticRegression(penalty='l1')
model3 = LogisticRegression(penalty='l1')


model1.fit(trn[columns1], y)
model2.fit(trn[columns2], y)

pp1 = model1.predict_proba(trn[columns1])[:, 1]
pp2 = model2.predict_proba(trn[columns2])[:, 1]

auc1 = roc_auc_score(y, pp1)
auc2 = roc_auc_score(y, pp2)

print auc1, auc2

p1 = model1.predict(trn[columns1])
p2 = model2.predict(trn[columns2])

scr1 = accuracy_score(y, p1)
scr2 = accuracy_score(y, p2)

print scr1, scr2

trn['pp1'] = pp1
trn['pp2'] = pp2

plt.subplot(211)
plt.plot(trn.sort_values(by='pp1', ascending=False)[target])
plt.title('pp1')
plt.subplot(212)
plt.plot(trn.sort_values(by='pp2', ascending=False)[target])
plt.title('pp2')
plt.show()




z_statistics2 = pd.Series()
for c in columns2:
	z = fa.Ztest_bi(trn, 'is_attributed', c)
	z_statistics2[c] = z

z_statistics2.sort_values(ascending=False, inplace=True)



dec_func1 = model1.decision_function(trn[columns1])
dec_func2 = model2.decision_function(trn[columns2])


trn['dec_func1'] = dec_func1
trn['dec_func2'] = dec_func2


trn.groupby('is_attributed')[['dec_func1', 'dec_func2']].describe()

z1 = fa.Ztest_bi(trn, 'is_attributed', 'dec_func1')
z2 = fa.Ztest_bi(trn, 'is_attributed', 'dec_func2')

from sklearn.metrics import confusion_matrix

c1 = confusion_matrix(y, p1)
c2 = confusion_matrix(y, p2)

print c1
print c2


from sklearn.metrics import log_loss

pp11 = model1.predict_proba(trn[columns1])
pp22 = model2.predict_proba(trn[columns2])

l1 = log_loss(y, pp11)
l2 = log_loss(y, pp22)

print l1, l2

from sklearn.metrics import roc_curve

fpr1, tpr1, thresholds1 = roc_curve(y, pp1)
fpr2, tpr2, thresholds2 = roc_curve(y, pp2)

plt.plot(fpr1, tpr1)
plt.plot(fpr2, tpr2)
plt.legend(['columns1', 'columns2'], loc='best')
plt.show()

columns3 = total_columns
columns3 = use_columns

## == 

columns1 = oof_columns
columns2 = smooth_columns
columns3 = stats_columns
columns4 = tgt_stats_columns

model1 = LogisticRegression(penalty='l1')
model2 = LogisticRegression(penalty='l1')
model3 = LogisticRegression(penalty='l1')
model4 = LogisticRegression(penalty='l1')


print cv_res1.mean(), cv_res1.std()


model = LogisticRegression(penalty='l1')
y_trn = trn[target]


use_columns = list()
use_columns += oof_columns
use_columns += smooth_columns
# use_columns += cate_freq_columns
# use_columns += stats_columns
use_columns += tgt_stats_columns


col_dict = list()
col_dict.append(('oof', oof_columns))
col_dict.append(('smooth', smooth_columns))
col_dict.append(('stats', stats_columns))
col_dict.append(('tgt_stats', tgt_stats_columns))
col_dict.append(('all', use_columns))

model = LogisticRegression(penalty='l1')
for name, columns in col_dict:
	cv_res = cross_val_score(model, trn[columns], trn[target], cv=5, scoring='roc_auc')
	# msg1 = "(%.3f, %.3f)" % (cv_res.mean(), cv_res.std())

	model.fit(trn[columns], trn[target])
	pp_trn = model.predict_proba(trn[columns])[:, 1]
	pp_tst = model.predict_proba(tst[columns])[:, 1]

	auc_trn = roc_auc_score(y_trn, pp_trn)
	auc_tst = roc_auc_score(y_tst, pp_tst)

	# print name, ': cv =', msg1, ", train =", auc_trn, ', test =', auc_tst
	print '{:>10}: cv = {}, train = {}, test = {}'.format(name, (cv_res.mean(), cv_res.std()), auc_trn, auc_tst)











