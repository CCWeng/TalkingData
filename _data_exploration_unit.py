import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


trn_spl = pd.read_csv('__input/train_sample.csv')
tst_sup = pd.read_csv('__input/test_supplement.csv')

tst = pd.read_csv('__input/test.csv')
sub = pd.read_csv('__input/sample_submission.csv')

# orig_columns = trn_spl.columns.tolist()
orig_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time']
target = 'is_attributed'


'''
====
ip
----
# of labels = 277396

====
'''



trn = pd.read_csv('__input/train.csv', usecols=['ip', 'is_attributed'])


ip_count = trn.groupby('ip').size()
ip_count = ip_count.to_frame('click_count')
ip_attributed = trn.groupby('ip')['is_attributed'].sum()
ip_attributed = ip_attributed.to_frame('download_count')

ip_stats = pd.concat([ip_count, ip_attributed], axis=1)


dc_dummies = pd.get_dummies(ip_stats.download_count, prefix='dc')

ip_stats2 = pd.concat([ip_stats, dc_dummies], axis=1).drop('download_count', axis=1)

gby = pd.concat([ip_stats2.groupby('click_count').mean(), ip_stats2.groupby('click_count').size().to_frame('counts')], axis=1)


gby.loc[gby.counts>100].iloc[:, :10].plot(kind='bar', stacked=True, figsize=(15, 5))
plt.show()



# ip appearance(counts)
# ip download ratio
# Separate data by ip appearance ?


'''
======
app
======
'''

trn = pd.read_csv('__input/train.csv', usecols=['app', 'is_attributed'])

apps = sorted(trn.app.unique())

app_counts = trn.groupby('app').size().to_frame('click_count')
app_attributed = trn.groupby('app')['is_attributed'].sum().to_frame('download_count')

app_stats = pd.concat([app_counts, app_attributed], axis=1)
app_stats['download_ratio'] = app_stats.download_count.astype(float) / app_stats.click_count

# Separate data by app appearance ?


'''
========
device
========
'''

trn = pd.read_csv('__input/train.csv', usecols=['device', 'is_attributed'])

devices = sorted(trn.device.unique())

dev_click = trn.groupby('device').size().to_frame('click_count')
dev_attributed = trn.groupby('device')['is_attributed'].sum().to_frame('download_count')

dev_stats = pd.concat([dev_click, dev_attributed], axis=1)
dev_stats['download_ratio'] = dev_stats.download_count.astype(float) / dev_stats.click_count


dev_stats.loc[dev_stats.click_count <= 18, 'download_ratio'].plot('density')
dev_stats.loc[dev_stats.click_count > 18, 'download_ratio'].plot('density')
plt.legend(['click_count<=18', 'click_count>18'], loc='best')
plt.title('download ratio')
plt.show()


# Separate data by click count <= 18



'''
====
os
====
'''

trn = pd.read_csv('__input/train.csv', usecols=['os', 'is_attributed'])

oss = sorted(trn.os.unique())

os_click = trn.groupby('os').size().to_frame('click_count')
os_attributed = trn.groupby('os')['is_attributed'].sum().to_frame('download_count')

os_stats = pd.concat([os_click, os_attributed], axis=1)
os_stats['download_ratio'] = os_stats.download_count.astype(float) / os_stats.click_count



quantile_levels = np.linspace(0.8, 1, 6)
download_levels = [os_stats.download_ratio.quantile(p) for p in quantile_levels]
download_levels = sorted(list(set(download_levels)))[1:]

def get_d_level(x):
	for i, v in enumerate(download_levels):
		if x <= v:
			return i


os_stats['download_level'] = os_stats.download_ratio.apply(get_d_level).astype(int)



dc_dummies = pd.get_dummies(os_stats.download_level, prefix='dl')

os_stats2 = pd.concat([os_stats, dc_dummies], axis=1).drop(['download_count', 'download_ratio', 'download_level'], axis=1)


gby = pd.concat([os_stats2.groupby('click_count').mean(), os_stats2.groupby('click_count').size().to_frame('counts')], axis=1)


gby.drop('counts', axis=1).plot(kind='bar', stacked=True, figsize=(15, 5))
plt.show()




attr_dummies = pd.get_dummies(trn.is_attributed, prefix='attr')
trn = pd.concat([trn, attr_dummies], axis=1).drop('is_attributed', axis=1)

gby = pd.concat([trn.groupby('os').mean(), trn.groupby('os').size().to_frame('click_count')], axis=1)
gby2 = gby.sort_values(by='click_count', ascending=False).head(150)

gby2.drop('click_count', axis=1).plot(kind='bar', stacked=True, figsize=(15, 5))
plt.show()


# os oof?


'''
=========
channel
=========
'''

trn = pd.read_csv('__input/train.csv', usecols=['channel', 'is_attributed'])

channels = sorted(trn.channel.unique())

dc_dummies = pd.get_dummies(trn.is_attributed)

trn = pd.concat([trn, dc_dummies], axis=1).drop('is_attributed', axis=1)
gby = pd.concat([trn.groupby('channel').mean(), trn.groupby('channel').size().to_frame('click_count')], axis=1)
gby2 = gby.sort_values(by='click_count', ascending=False).head(100)

gby2.drop('click_count', axis=1).plot(kind='bar', stacked=True, figsize=(15, 5))
plt.show()


## channel based? oof?

'''
=================
attributed_time
=================
'''

trn = pd.read_csv('__input/train.csv', usecols=['attributed_time', 'is_attributed'])

# not used
# download speed of each channel? ip? device?


'''
============
click_time
============
'''

trn = pd.read_csv('__input/train.csv', usecols=['click_time', 'is_attributed'])
tst = pd.read_csv('__input/test.csv', usecols=['click_time'])


click_time = pd.to_datetime(trn.click_time)

# trn['year'] = click_time.dt.year # only 2017
# trn['quarter'] = click_time.dt.quarter
# trn['month'] = click_time.dt.month
trn['day'] = click_time.dt.day
# trn['week'] = click_time.dt.week
trn['dow'] = click_time.dt.dayofweek
# trn['doy'] = click_time.dt.dayofyear


grp = trn.groupby(['day', 'is_attributed']).size()
rat = grp.loc[:, 1, :].astype(float) / grp.loc[:, 0, :]

trn['hour'] = click_time.dt.hour

grp_hr = trn.groupby(['hour', 'is_attributed']).size()
rat_hr = grp_hr.loc[:, 1, :].astype(float) / grp_hr.loc[:, 0, :]


days = sorted(trn.day.unique())

for d in days:
	grp = trn.loc[trn.day==d].groupby(['hour', 'is_attributed']).size()
	rat = grp.loc[:, 1, :].astype(float) / grp.loc[:, 0, :]
	plt.plot(rat.index.tolist(), rat)


plt.legend(["day=%d" % d for d in days], loc='best')
plt.show()

grp_hr = trn.groupby(['day', 'hour', 'is_attributed']).size()
rat_hr = grp_hr.loc[:, :, 1, :].astype(float) / grp_hr.loc[:, :, 0, :]



def get_time_interval(x):
	time_interval = 1 if x > 12 else 0
	# time_interval = 1 if 6 < x and x <= 18 else 0
	return time_interval

trn['time_interval'] = trn.hour.apply(get_time_interval)


# Train interval : 2017-11-06 ~ 2017-11-09
# Test interval : 2017-11-10 04 ~ 2017-11-10 15

# categorical by hour
# tendency by hour








