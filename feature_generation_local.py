import pandas as pd
import numpy as np


def CreateClicksColumns(trn, tst, secondary_group_indice=[]):
	trn['train'] = 1
	tst['train'] = 0
	trn['row_id'] = range(len(trn))
	tst['row_id'] = range(len(tst))

	if 'ip' not in secondary_group_indice:
		secondary_group_indice.append('ip')
	
	use_columns = secondary_group_indice + ['channel', 'train', 'row_id']
	df_all = pd.concat([trn[use_columns], tst[use_columns]], axis=0, ignore_index=False)

	
	new_columns = list()
	for gp in secondary_group_indice:
		if gp == 'ip':
			new_c = 'clicks_by_ip'
			ip_count = df_all.groupby(['ip'])['channel'].count().reset_index()
		else:
			new_c = 'clicks_by_ip&' + gp
			ip_count = df_all.groupby(['ip', gp])['channel'].count().reset_index()
	
		new_columns.append(new_c)
		ip_count = ip_count.rename(columns={'channel': new_c})

		df_all = pd.merge(df_all, ip_count, on=['ip', gp], how='left', sort=False)
		df_all[new_c] = df_all[new_c].astype('uint16')
	
	df_all.drop('ip', axis=1, inplace=True)

	clicks_trn = df_all.loc[df_all.train==1].sort_values(by='row_id').set_index(trn.index)
	clicks_tst = df_all.loc[df_all.train==0].sort_values(by='row_id').set_index(tst.index)

	trn[new_columns] = clicks_trn[new_columns]
	tst[new_columns] = clicks_tst[new_columns]
	# trn = pd.concat([trn, clicks_trn[new_columns]], axis=1)
	# tst = pd.concat([tst, clicks_tst[new_columns]], axis=1)

	trn.drop(['train', 'row_id'], axis=1, inplace=True)
	tst.drop(['train', 'row_id'], axis=1, inplace=True)

	return trn, tst, new_columns

