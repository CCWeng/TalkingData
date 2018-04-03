from sklearn.linear_model import LogisiticRegression


model_lr = LogisiticRegression(penalty='l1')

model_lr.fit(trn[use_columns], trn[target])
pp = model.predict_proba(tst[use_columns])[:, 1]

tst['is_attributed'] = pp

d_out = '__output/'
output_columns = ['click_id', 'is_attributed']

tst.to_csv(d_output+'lr_180403_01.csv', columns = output_columns, index=False)