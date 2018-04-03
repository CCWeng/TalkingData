import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


trn = pd.read_csv('__input/train.csv')


ip_count = trn.groupby('ip').size()
ip_count = ip_count.to_frame('click_count')
ip_attributed = trn.groupby('ip')['is_attributed'].sum()
ip_attributed = ip_attributed.to_frame('download_count')

ip_stats = pd.concat([ip_count, ip_attributed], axis=1)


dc_dummies = pd.get_dummies(ip_stats.download_count, prefix='dc')

ip_stats2 = pd.concat([ip_stats, dc_dummies], axis=1).drop('download_count', axis=1)

gby = pd.concat([ip_stats2.groupby('click_count').mean(), ip_stats2.groupby('click_count').count()], axis=1).iloc[:, :-2]
gby.columns = ['dc_0', 'dc_1', 'dc_3', 'count']
