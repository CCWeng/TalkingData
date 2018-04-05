import pandas as pd
import numpy as np
import pdb
import gc

from sklearn.model_selection import train_test_split

import feature_generation as fg
import feature_analysis as fa


f_trn = '__input/train.csv'
f_tst = '__input/test.csv'

trn_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
tst_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}

resize_data = False

if not resize_data:
	n_skiprows = 110000000

	trn = pd.read_csv(f_trn, dtype=dtypes, usecols=trn_cols, skiprows=range(1, n_skiprows))
	tst = pd.read_csv(f_tst, dtype=dtypes, usecols=tst_cols)
else:
	trn = pd.read_csv(f_trn, dtype=dtypes, usecols=trn_cols)

	dat, _ = train_test_split(trn, train_size=0.003, test_size=0.0001)
	del trn, _

	gc.collect()

	tst_datetime = '2017-11-09 00:00:00'
	index_trn = dat.loc[dat.click_time < tst_datetime].index
	index_tst = dat.loc[dat.click_time >= tst_datetime].index

	trn = dat.loc[index_trn]
	tst = dat.loc[index_tst]

	del dat
	gc.collect()

print '\nSize of train data :', trn.shape, '\nSize of test data :', tst.shape



# trn, tst = train_test_split(trn, train_size=0.0007, test_size=0.0003)
# X_trn, X_tst, y_trn, y_tst = train_test_split(trn[])

target = 'is_attributed'

orig_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time']
cate_columns = ['ip', 'app', 'device', 'os', 'channel']
dumm_columns = list()



'''== hour of click time =='''

click_time = pd.to_datetime(trn.click_time)
trn['click_hour'] = click_time.dt.hour

click_time = pd.to_datetime(tst.click_time)
tst['click_hour'] = click_time.dt.hour

del click_time
gc.collect()

cate_columns.append('click_hour')


'''=== OOF features ==='''

trn, tst, oof_columns = fg.CreateOOFColumns(trn, tst, cate_columns, target=target)

# trn.groupby('is_attributed')[oof_columns].describe()

# for ns in np.linspace(0, 0.2, 10):
# # for ns in np.arange(0, 0.5, 0.1):
# 	trn, tst, oof_columns = fg.CreateOOFColumns(trn, tst, cate_columns, target='is_attributed', noise_level=0.1, verbose=False)
# 	res = cross_val_score(model_lr, trn[oof_columns], trn[target], cv=5, scoring='roc_auc')
# 	print ns, ':', res.mean(), ',', res.std()



'''== smoothing features =='''

trn, tst, smooth_columns = fg.CreateSmoothingColumns(trn, tst, cate_columns, target=target)

# trn.groupby('is_attributed')[smooth_columns].describe()


# for ns in np.arange(0, 0.5, 0.1):
# 	trn, tst, smooth_columns = fg.CreateSmoothingColumns(trn, tst, cate_columns, target='is_attributed', noise_level=0.1, verbose=False)
# 	res = cross_val_score(model_lr, trn[smooth_columns], trn[target], cv=5, scoring='roc_auc')
# 	print ns, ':', res.mean(), ',', res.std()




'''== dummy columns for click hour =='''


# dumm = pd.get_dummies(trn.click_hour, prefix='click_hour')
# trn = pd.concat([trn, dumm], axis=1)

# dumm = pd.get_dummies(tst.click_hour, prefix='click_hour')
# tst = pd.concat([tst, dumm], axis=1)

# dumm_columns += dumm.columns.tolist()


# trn, tst, oof_hour_columns = fg.CreateOOFColumns(trn, tst, ['click_hour'], target='is_attributed')
# trn, tst, smooth_hour_columns = fg.CreateSmoothingColumns(trn, tst, ['click_hour'], target='is_attributed')


## oof = smooth > dummy




'''== Rescale smooth columns =='''
# from sklearn.preprocessing import StandardScaler

# rescaled_smooth_columns = ['rescaled_' + c for c in smooth_columns]

# scaler = StandardScaler()
# scaler = scaler.fit(trn[smooth_columns])

# trn_mat = scaler.transform(trn[smooth_columns])
# trn_new = pd.DataFrame(trn_mat, columns=rescaled_smooth_columns, index=trn.index)
# trn = pd.concat([trn, trn_new], axis=1)

# tst_mat = scaler.transform(tst[smooth_columns])
# tst_new = pd.DataFrame(tst_mat, columns=rescaled_smooth_columns, index=tst.index)
# tst = pd.concat([tst, tst_new], axis=1)



# trn, tst, rescaled_dumm_hour_columns = fg.CreateRescaledColumns(trn, tst, dumm_columns)


## rescaling almost change nothing



'''== categorical appearance level features =='''


trn, tst, cate_freq_columns = fg.CreateCateFreqColumns(trn, tst, cate_columns)





'''== statistics of 'is_attributed' on categorical_columns =='''

# trn, tst, stats_columns = fg.CreateStatsFeatures(trn, tst, observe_columns=[target], group_columns=cate_columns)


# del_columns = list()
# for c in stats_columns:
# 	if len(trn[c].unique()) == 1:
# 		del_columns.append(c)

# for c in del_columns:
# 	stats_columns.remove(c)

# z_statistics = pd.Series()
# for c in stats_columns:
# 	z = fa.Ztest(trn, 'is_attributed', c)
# 	z_statistics[c] = z


# z_statistics.sort_values(ascending=False, inplace=True)


# stats_columns2 = z_statistics.index.tolist()

# for i in range(1, len(stats_columns2)+1):
# 	add_columns = stats_columns2[:i]


## information leak!


## categorical stats columns



## delta time(attributed_time - click_time) 

## number of clicks on app/ip/os

## time series based features


## dim-2 combined categorical features
## - do data exploration first


## Define total columns in utility

use_columns = list()
use_columns += oof_columns
use_columns += smooth_columns
use_columns += cate_freq_columns
# use_columns += stats_columns



# tmp

from sklearn.linear_model import LogisiticRegression
from sklearn.model_selection import cross_val_score


model_lr = LogisiticRegression(penalty='l1')

cv_res = cross_val_score(model_lr, trn[use_columns], trn[target], cv=5, scoring='roc_auc')
print 'Results of cross-validation :', cv_res.mean(), cv_res.std()

model_lr.fit(trn[use_columns], trn[target])
pp = model.predict_proba(tst[use_columns])[:, 1]

tst['is_attributed'] = pp

d_out = '__output/'
output_columns = ['click_id', 'is_attributed']

tst.to_csv(d_output+'lr_180405_01.csv', columns = output_columns, index=False)




