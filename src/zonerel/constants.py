LOG_COLS=['GR','AC','DEN','CNL','RT']
NUM_COLS=['GR','AC','DEN','CNL','RT','pressure_coeff','formation_top_md','formation_bottom_md','max_burial_top']
CAT_COLS=['layer','lithology','diagenesis_category']
MODEL_ORDER=['train_mean','zone_mean','LinearRegression','Ridge','Lasso','ElasticNet','Huber','SVR_RBF','KNN','RF','ExtraTrees','GBDT','HistGBDT','XGBoost','LightGBM','CatBoost']
FAST_MODELS=['train_mean','zone_mean','Ridge','RF','ExtraTrees']
ZONE_ORDER=['North','Central','South']
ZONE_EN={'\u5317':'North','\u4e2d':'Central','\u5357':'South','North':'North','Central':'Central','South':'South'}
