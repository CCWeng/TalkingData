import pandas as pd
import numpy as np
import sys
import gc


def CreateTimeDiffColumns(trn, tst, groupbys):
	index = trn.loc[trn.is_attributed == 1].index
	t1 = pd.to_datetime(trn.loc[index, 'click_time'])
	t2 = pd.to_datetime(trn.loc[index, 'attributed_time'])

	trn['time_diff'] = np.nan
	trn.loc[index, 'time_diff'] = t2.values - t1.values

	trn['sec_diff'] = np.nan
	trn.loc[index, 'sec_diff'] = trn.loc[index, 'time_diff'].apply(lambda x: x.seconds)

	trn['min_diff'] = np.nan
	trn.loc[index, 'min_diff'] = (trn.loc[index, 'sec_diff'] / 60).round()


	total_new_columns = list()
	for c in groupbys:
		sys.stdout.write("Create time difference for %s ... " % c)
		sys.stdout.flush()

		new_cols = [c+'_mindiff', c+'_secdiff']
		total_new_columns += new_cols

		g1 = trn.loc[index].groupby(c)['min_diff'].mean()
		g2 = trn.loc[index].groupby(c)['sec_diff'].mean()

		g = pd.concat([g1, g2], axis=1)
		g.columns = new_cols

		trn = trn.join(g, on=c, how='left')
		tst = tst.join(g, on=c, how='left')

	trn.drop(['time_diff', 'min_diff', 'sec_diff'], axis=1, inplace=True)
	gc.collect()

	return trn, tst, total_new_columns







def CreateClickCntColumns_Multi(trn, tst, groupbys):
	print "Compute click counts :"

	trn['train'] = 1
	tst['train'] = 0
	trn['row_id'] = range(len(trn))
	tst['row_id'] = range(len(tst))

	use_columns = set()
	for tup in groupbys:
		use_columns = use_columns.union(tup)

	use_columns = list(use_columns)
	use_columns += ['train', 'row_id']

	df_all = trn[use_columns].append(tst[use_columns])

	N = len(groupbys)
	cnt_columns = list()
	inv_columns = list()
	for i, gb in enumerate(groupbys):
		c = '&'.join(gb)
		sys.stdout.write("  %s (%d/%d) ... " % (c, i+1, N))
		sys.stdout.flush()

		cnt_c = c + '_clickcnt'
		cnt_columns.append(cnt_c)
		
		cnt = df_all.groupby(gb).size().reset_index()
		cnt.columns = (gb + [cnt_c])

		df_all = df_all.reset_index()
		df_all = pd.merge(df_all, cnt, on=gb, how='left')
		df_all = df_all.set_index('index')

		# inv_c = c + '_clickcnt_inv'
		# inv_columns.append(inv_c)
		# df_all[inv_c] = 1. / df_all[cnt_c]

		df_all[cnt_c] = df_all[cnt_c].astype('uint16')

		sys.stdout.write("done.\n")

	trn_new = df_all.loc[df_all.train==1].sort_values(by='row_id')
	tst_new = df_all.loc[df_all.train==0].sort_values(by='row_id')

	del df_all
	gc.collect()

	new_columns = cnt_columns + inv_columns
	for c in new_columns:
		if c in trn:
			trn.drop(c, axis=1, inplace=True)
		if c in tst:
			tst.drop(c, axis=1, inplace=True)

	trn = pd.concat([trn, trn_new[new_columns]], axis=1)
	tst = pd.concat([tst, tst_new[new_columns]], axis=1)

	trn.drop(['train', 'row_id'], axis=1, inplace=True)
	tst.drop(['train', 'row_id'], axis=1, inplace=True)

	return trn, tst, cnt_columns #, inv_columns



def CreateClickCntColumns(trn, tst, cate_columns):
	print "Compute click counts :"

	trn['train'] = 1
	tst['train'] = 0
	trn['row_id'] = range(len(trn))
	tst['row_id'] = range(len(tst))

	use_columns = cate_columns + ['train', 'row_id']
	df_all = trn[use_columns].append(tst[use_columns])

	N = len(cate_columns)
	cnt_columns = list()
	inv_columns = list()
	for i, c in enumerate(cate_columns):
		sys.stdout.write("  %s (%d/%d) ... " % (c, i+1, N))
		sys.stdout.flush()

		cnt = df_all.groupby(c).size()

		cnt_c = c + '_clickcnt'
		cnt_columns.append(cnt_c)
		df_all[cnt_c] = df_all[c].map(cnt)
		
		inv_c = c + '_clickcnt_inv'
		inv_columns.append(inv_c)
		df_all[inv_c] = 1. / df_all[cnt_c]

		df_all[cnt_c] = df_all[cnt_c].astype('uint16')

		sys.stdout.write("done.\n")

	trn_new = df_all.loc[df_all.train==1].sort_values(by='row_id')
	tst_new = df_all.loc[df_all.train==0].sort_values(by='row_id')

	del df_all
	gc.collect()

	new_columns = cnt_columns + inv_columns
	for c in new_columns:
		if c in trn:
			trn.drop(c, axis=1, inplace=True)
		if c in tst:
			tst.drop(c, axis=1, inplace=True)

	trn = pd.concat([trn, trn_new[new_columns]], axis=1)
	tst = pd.concat([tst, tst_new[new_columns]], axis=1)

	trn.drop(['train', 'row_id'], axis=1, inplace=True)
	tst.drop(['train', 'row_id'], axis=1, inplace=True)

	return trn, tst, cnt_columns, inv_columns


def CreateAttrCumColumns(trn, tst, groupbys, use_noise=False):
	print "Compute cumulative attributed numbers :"

	prior = trn.is_attributed.mean()

	trn['train'] = 1
	tst['train'] = 0
	trn['row_id'] = range(len(trn))
	tst['row_id'] = range(len(tst))
	
	trn['is_attributed2'] = trn.is_attributed
	tst['is_attributed2'] = prior

	if use_noise:
		std = trn.is_attributed.std()
		noise_level = std
		trn['is_attributed2'] += (noise_level * np.random.randn(len(trn)))
		tst['is_attributed2'] += (noise_level * np.random.randn(len(tst)))

	use_columns = set()
	for tup in groupbys:
		use_columns = use_columns.union(tup)

	use_columns = list(use_columns)
	use_columns += ['is_attributed2', 'train', 'row_id', 'click_time']

	df_all = trn[use_columns].append(tst[use_columns])
	df_all.sort_values(by='click_time', inplace=True)

	new_columns = list()
	N = len(groupbys)
	for i, gb in enumerate(groupbys):
		c = '&'.join(gb)
		sys.stdout.write("  %s (%d/%d) ... " % (c, i+1, N))
		sys.stdout.flush()

		attr_c = c + '_attrcum'
		new_columns.append(attr_c)

		df_all[attr_c] = df_all.is_attributed2
		df_all[attr_c] = df_all.groupby(gb)[attr_c].apply(lambda x: x.cumsum()-x)

		sys.stdout.write('done.\n')
		
	trn_new = df_all.loc[df_all.train==1].sort_values(by='row_id')
	tst_new = df_all.loc[df_all.train==0].sort_values(by='row_id')

	for c in new_columns:
		if c in trn:
			trn.drop(c, axis=1, inplace=True)
		if c in tst:
			tst.drop(c, axis=1, inplace=True)

	trn = pd.concat([trn, trn_new[new_columns]], axis=1)
	tst = pd.concat([tst, tst_new[new_columns]], axis=1)

	trn.drop(['train', 'row_id', 'is_attributed2'], axis=1, inplace=True)
	tst.drop(['train', 'row_id', 'is_attributed2'], axis=1, inplace=True)

	return trn, tst, new_columns





def CreateClickCumColumns(trn, tst, groupbys):
	print "Compute cumulative clicks numbers :"

	trn['train'] = 1
	tst['train'] = 0
	trn['row_id'] = range(len(trn))
	tst['row_id'] = range(len(tst))

	use_columns = set()
	for tup in groupbys:
		use_columns = use_columns.union(tup)

	use_columns = list(use_columns)
	use_columns += ['train', 'row_id', 'click_time']

	df_all = trn[use_columns].append(tst[use_columns])
	df_all.sort_values(by='click_time', inplace=True)

	new_columns = list()
	N = len(groupbys)
	for i, gb in enumerate(groupbys):
		c = '&'.join(gb)
		sys.stdout.write("  %s (%d/%d) ... " % (c, i+1, N))
		sys.stdout.flush()

		click_c = c + '_clickcum'
		new_columns.append(click_c)

		df_all[click_c] = 1
		df_all[click_c] = df_all.groupby(gb)[click_c].apply(lambda x: x.cumsum())

		sys.stdout.write('done.\n')
		
	trn_new = df_all.loc[df_all.train==1].sort_values(by='row_id')
	tst_new = df_all.loc[df_all.train==0].sort_values(by='row_id')

	for c in new_columns:
		if c in trn:
			trn.drop(c, axis=1, inplace=True)
		if c in tst:
			tst.drop(c, axis=1, inplace=True)

	trn = pd.concat([trn, trn_new[new_columns]], axis=1)
	tst = pd.concat([tst, tst_new[new_columns]], axis=1)

	trn.drop(['train', 'row_id'], axis=1, inplace=True)
	tst.drop(['train', 'row_id'], axis=1, inplace=True)

	return trn, tst, new_columns




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

