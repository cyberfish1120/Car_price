# --- coding:utf-8 ---
# author: Cyberfish time:2020/7/29

import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn_zeros  # 有问题
from IPython.display import display, clear_output

warnings.filterwarnings('ignore')

# 模型预测工具
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 数据降维处理工具
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, SparsePCA
import lightgbm as lgb
import xgboost as xgb

# 参数搜索和评价工具
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error

Train_data = pd.read_csv('../data/used_car_train_20200313.csv', sep=' ')
TestB_data = pd.read_csv('../data/used_car_testB_20200421.csv', sep=' ')

# 提取数值型特征列
numberical_cols = Train_data.select_dtypes(exclude='object').columns
# print(numberical_cols)
categorical_cols = Train_data.select_dtypes(include='object').columns

# 选择特征列
feature_cols = [col for col in numberical_cols if col not in
                ['SaleID', 'name', 'regDate', 'creatDate', 'price',
                 'model', 'brand', 'regionCode', 'seller']]
feature_cols = [col for col in feature_cols if 'Type' not in col]

# 提前特征列，标签列构造训练样本和测试样本
X_data = Train_data[feature_cols]  # 150000x18
Y_data = Train_data['price']  # 150000x1
X_test = TestB_data[feature_cols]  # 50000x18


# print('X train shape: ', X_data.shape)


def sta_inf(data):
    """定义一个统计函数，方便后续信息统计"""
    print('_min', np.min(data))
    print('_max:', np.max(data))
    print('_mean', np.mean(data))
    print('_ptp', np.ptp(data))
    print('_std', np.std(data))
    print('_var', np.var(data))


# 缺省值用-1填补
X_data = X_data.fillna(-1)
Y_data = Y_data.fillna(-1)

'''**********************************
              模型训练与预测
   ********************************** '''
# 利用xgb进行五折交叉验证查看模型的参数效果
xgr = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, gamma=0,
                       subsample=0.8, colsample_bytree=0.9, max_depth=7)
scores_train, scores = [], []
# 五折交叉验证方式
sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_ind, val_ind in sk.split(X_data, Y_data):
    train_x = X_data.iloc[train_ind].values
    train_y = Y_data.iloc[train_ind]
    val_x = X_data.iloc[val_ind].values
    val_y = Y_data.iloc[val_ind]

    xgr.fit(train_x, train_y)
    pred_train_xgb = xgr.predict(train_x)
    pred_xgb = xgr.predict(val_x)

    score_train = median_absolute_error(train_y, pred_train_xgb)
    scores_train.append(score_train)
    score = median_absolute_error(val_y, pred_xgb)
    scores.append(score)
print('Train mae: ', np.mean(scores_train))
print('Val mae: ', np.mean(scores))


# 定义xgb和lgb模型函数
def build_model_xgb(x_train, y_train):
    model = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, gamma=0,
                             subsample=0.8, colsample_bytree=0.9, max_depth=7)
    model.fit(x_train, y_train)
    return model


def build_model_lgb(x_train, y_train):
    estimator = lgb.LGBMRegressor(num_leaves=127, n_estimators=150)
    param_grid = {'learning_rate': [0.01, 0.05, 0.1, 0.2]}
    gbm = GridSearchCV(estimator, param_grid)
    gbm.fit(x_train, y_train)
    return gbm


# 切分数据集，70%训练，30%验证
x_train, x_val, y_train, y_val = train_test_split(X_data, Y_data, test_size=0.3)

print('###############Train lgb###############')  # 为了后续计算两个模型的权重
model_lgb = build_model_lgb(x_train, y_train)
val_lgb = model_lgb.predict(x_val)
MAE_lgb = median_absolute_error(y_val, val_lgb)
print('mean lgb: ', MAE_lgb)
print('############Predict lgb###############')
model_lgb_pre = build_model_lgb(X_data, Y_data)
test_lgb = model_lgb_pre.predict(X_test)
print('Sta of Predict xgb:')
sta_inf(test_lgb)

print('##############Train xgb###############')  # 为了后续计算两个模型的权重
model_xgb = build_model_xgb(x_train, y_train)
val_xgb = model_xgb.predict(x_val)
MAE_xgb = median_absolute_error(y_val, val_xgb)
print('mean xgb: ', MAE_xgb)
print('##############Prectict xgb#############')
model_xgb_pre = build_model_xgb(X_data, Y_data)
test_xgb = model_xgb_pre.predict(X_test)
print('Sta of Predict xgb:')
sta_inf(test_xgb)

# 两个模型进行加权融合
test_w = (MAE_xgb/(MAE_lgb + MAE_xgb)) * test_lgb + (MAE_lgb/(MAE_lgb + MAE_xgb)) * test_xgb
test_w[test_w < 0] = 10

# 输出结果
ans = pd.DataFrame()
ans['SaleID'] = TestB_data['SaleID']
ans['price'] = test_w
ans.to_csv('../ans.csv', index=False)

print('*********************************\n', ans.describe())
