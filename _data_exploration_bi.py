import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

f_trn = '__input/train.csv'
f_tst = '__input/test.csv'

orig_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time']
target = 'is_attributed'


# two-way stacked bar
# chi-square


for c in ['ip', 'app', 'device', 'os', 'channel']:
	dat = pd.read_csv(f_trn, usecols=[c, target])
	n_label = len(dat[c].unique())
	chi2_val, p_val = chi2(dat[[c]], dat[target])
	msg = "Feature %s: number of labels = %d, (chi2, pval) with target = (%.3f, %.3f)" % (c, n_label, chi2_val[0], p_val[0])
	print msg

'''
Feature ip: number of labels = 277396, (chi2, pval) with target = (28504755752.716, 0.000)
Feature app: number of labels = 706, (chi2, pval) with target = (12035402.008, 0.000)
Feature device: number of labels = 3475, (chi2, pval) with target = (2146593.290, 0.000)
Feature os: number of labels = 800, (chi2, pval) with target = (66128.973, 0.000)
Feature channel: number of labels = 202, (chi2, pval) with target = (6733436.458, 0.000)
'''


for c in ['ip', 'app', 'device', 'os', 'channel']:
	data = pd.read_csv(f_trn, usecols=[c, target])
	n_label = len(data[c].unique())
	chi2_val, p_val = chi2(data[[c]], data[target])
	msg = "Feature %s: number of labels = %d, (chi2, pval) with target = (%.3f, %.3f)" % (c, n_label, chi2_val[0], p_val[0])
	print msg





'''
===========
ip & app
===========
'''

col1 = 'ip'
col2 = 'app'

trn = pd.read_csv('__input/train.csv', usecols=[col1, col2, target])

X, _, y, _ = train_test_split(trn[[col1, col2]], trn[target], train_size=0.3)

chi2, pval = chi2(X[[col1]] , X[col2])


chi2, pval = chi2(trn.ip, trn.app)


grp1 = trn.groupby([col1, col2, target]).size()
grp2 = trn.groupby([col1, col2]).size()

rat = grp.loc[:, :, 1, :].astype(float) / grp.loc[:, :, 0, :]
cnt = 


'''
==============
device & app
==============
'''

trn = pd.read_csv('__input/train.csv', usecols=['device', 'app'])

X, _, y, _ = train_test_split(trn[['device']], trn['app'], train_size=0.01)

chi2_val, p_val = chi2(X, y)





'''
==============
os & app
==============
'''

trn = pd.read_csv('__input/train.csv', usecols=['os', 'app'])

X, _, y, _ = train_test_split(trn[['os']], trn['app'], train_size=0.01)

chi2_val, p_val = chi2(X, y)



'''
=================================
click_time & attributed_time
=================================
'''


trn = pd.read_csv('__input/train.csv', usecols=['app', 'click_time', 'attributed_time', 'is_attributed'])

grp1 = trn.groupby('app')['is_attributed'].mean()

index = trn.loc[trn.is_attributed==1].index

click_time = pd.to_datetime(trn.loc[index, 'click_time'])
attributed_time = pd.to_datetime(trn.loc[index, 'attributed_time'])
delta_time = attributed_time - click_time

# df_time = pd.concat([click_time, attributed_time], axis=1)
# df_time['hour_comp'] = delta_time.apply(lambda x: x.components.hours)
# df_time['hour_total'] = delta_time.apply(lambda x: x / np.timedelta64(1, 'h'))

trn.loc[index, 'delta_hours'] = delta_time.apply(lambda x: x / np.timedelta64(1, 'h'))
grp2 = trn.loc[index].groupby('app')['delta_hours'].mean()

df_tmp = pd.concat([grp1, grp2], axis=1)
df_tmp = df_tmp.loc[df_tmp.is_attributed!=0]

df_tmp.plot(x='delta_hours', y='is_attributed', kind='scatter')

df_tmp['delta_hours_trans'] = df_tmp.delta_hours.apply(lambda x: np.exp(-x))
df_tmp.plot(x='delta_hours_trans', y='is_attributed', kind='scatter')







