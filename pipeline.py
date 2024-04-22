import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from transformers import ColumnDropper, InsertNa, CountNa, SumCols

with open('data/cols_to_drop.json', 'r') as file:
    cols_to_drop = json.load(file)

with open('data/prefix_mapper.json', 'r') as file:
    prefix_mapper = json.load(file)

cols_to_drop_ = cols_to_drop['constant'] + cols_to_drop['duplicate'] + ['ID']
saldo_columns = prefix_mapper['saldo']
imp_columns = prefix_mapper['imp']
delta_columns = prefix_mapper['delta']
num_columns = prefix_mapper['num']
ind_columns = prefix_mapper['ind']

pipe = Pipeline([
    ('column_dropper', ColumnDropper(columns=cols_to_drop_)),

    ('insert_na_saldo', InsertNa(columns=saldo_columns, value=0)),
    ('insert_na_imp', InsertNa(columns=imp_columns, value=0)),
    ('insert_na_delta', InsertNa(columns=delta_columns, value=9999999999)),
    ('insert_na_var3', InsertNa(columns=['var3'], value=-999999)),

    ('count_na_saldo', CountNa(prefix='saldo', columns=saldo_columns)),
    ('count_na_imp', CountNa(prefix='imp', columns=imp_columns)),
    ('count_na_delta', CountNa(prefix='delta', columns=delta_columns)),

    ('sum_saldo', SumCols(prefix='saldo', columns=saldo_columns)),
    ('sum_imp', SumCols(prefix='imp', columns=imp_columns)),
    ('sum_delta', SumCols(prefix='delta', columns=delta_columns)),
    ('sum_num', SumCols(prefix='num', columns=num_columns)),
    ('sum_ind', SumCols(prefix='ind', columns=ind_columns)),

    ('classifier', HistGradientBoostingClassifier(categorical_features=["var36", "var21"]))
])