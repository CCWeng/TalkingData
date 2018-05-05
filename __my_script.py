import pandas as pd
import numpy as np
import pdb
import gc

from sklearn.model_selection import train_test_split

import feature_generation as fg
import feature_analysis as fa
import feature_generation_local as fg2


f_trn = '__input/train.csv'
f_tst = '__input/test_supplement.csv'

trn_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed', 'attributed_time']
tst_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']

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

	y_tst = tst['is_attributed']
	tst.drop('is_attributed', axis=1, inplace=True)

	del dat
	gc.collect()


trn.sort_values(by='click_time', inplace=True)
tst.sort_values(by='click_time', inplace=True)

print '\nSize of train data :', trn.shape, '\nSize of test data :', tst.shape



target = 'is_attributed'

orig_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time']
cate_columns = ['ip', 'app', 'device', 'os', 'channel']
dumm_columns = list()



'''== hour of click time =='''

click_time = pd.to_datetime(trn.click_time)
trn['click_day'] = click_time.dt.day
trn['click_hour'] = click_time.dt.hour

click_time = pd.to_datetime(tst.click_time)
tst['click_day'] = click_time.dt.day
tst['click_hour'] = click_time.dt.hour

del click_time
gc.collect()

cate_columns.append('click_hour')


'''=== OOF features ==='''

# trn, tst, oof_columns = fg.CreateOOFColumns(trn, tst, cate_columns, target=target)


'''== smoothing features =='''

# trn, tst, smooth_columns = fg.CreateCvSmoothingColumns(trn, tst, cate_columns, target=target)


'''== Rescale smooth columns =='''

'''== categorical appearance level features =='''


# trn, tst, cate_freq_columns = fg.CreateCateFreqColumns(trn, tst, cate_columns)


'''== statistics of 'is_attributed' on categorical_columns =='''

# trn, tst, stats_columns = fg.CreateTargetStatsFeatures(trn, tst, cate_columns, target, ts_split=True)


'''== Create click columns =='''
# trn, tst, click_columns = fg2.CreateClicksColumns(trn, tst, ['app'])


'''=== Create 2-way categorical columns =='''
# pairs = [('ip', 'app'), ('ip', 'click_hour'), ('app', 'click_hour')]
# trn, tst, cate_2way_columns = fg.Create2WayCateColumns(trn, tst, pairs)


'''=== Create create count columns ==='''
trn, tst, clkcnt_columns, clkcnt_inv_columns = fg2.CreateClickCntColumns(trn, tst, cate_columns)

'''== Create click counts for multidimensional categorical features. ==''' 

# groupbys = list()
# groupbys.append(['ip', 'app'])
# groupbys.append(['ip', 'click_day', 'click_hour'])
# groupbys.append(['ip', 'app', 'os'])

# trn, tst, clkcnt_mul_columns, clkcnt_mul_inv_columns = fg2.CreateClickCntColumns_Multi(trn, tst, groupbys)





use_columns = list()
use_columns += cate_columns
use_columns += clkcnt_columns

# use_columns += oof_columns
# use_columns += smooth_columns
# use_columns += cate_freq_columns
# use_columns += stats_columns
# use_columns += stats_columns
# use_columns += click_columns
# use_columns += cate_2way_columns
# use_columns += clkcnt_inv_columns

# use_columns += clkcnt_mul_columns
# use_columns += clkcnt_mul_inv_columns
# use_columns += clkcnt_mul_cap_columns

# use_columns += clkcnt_mul_columns[:1]
# use_columns += clkcnt_mul_inv_columns[:1]
# use_columns += clkcnt_mul_cap_columns[:1]








# == 





