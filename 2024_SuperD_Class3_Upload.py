# %% [markdown]
# # Import module

# %%
#### Set to 1 if you want to perform hyperparmeter learning #####
hyper_para_learn = 0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import jaconv
import re
import pickle as pkl

# Change fontype of pyplot
import matplotlib.font_manager
#print([f.name for f in matplotlib.font_manager.fontManager.ttflist])
plt.rcParams['font.family'] = 'MS Gothic'
# For mac user
#plt.rcParams['font.family'] = 'AppleGothic'

# %%


# %% [markdown]
# # Load Data

# %%
df = pd.read_csv("SuperD_Class2_tochi_sample.csv")

# %%
# load code data
df_code = pd.read_csv("SuperD_Class2_shicyoukuson_code_utf8.csv")
# create dict
code2lng, code2lat = ({} for _ in range(2))
for i in range(len(df_code)):
    code = str(df_code["コード"].iloc[i])
    if (len(code) == 5):
        code = code[0:4]
    else:
        code = code[0:5]
    code2lng.update({int(code): df_code["経度"].iloc[i]})
    code2lat.update({int(code): df_code["緯度"].iloc[i]})
# longitude and latitude
df["longitude"] = df["市区町村コード"].map(code2lng)
df["latitude"] = df["市区町村コード"].map(code2lat)

# %%


# %% [markdown]
# # Price distribution

# %%
# too heavy tailed
df["取引価格（総額）"].hist(bins=50)
plt.title("Histogram of price")
plt.show()

# %%
# summary statistics
df["取引価格（総額）"].describe()

# %%
# sort
df.sort_values(by="取引価格（総額）",ascending=False,inplace=True)

# %%
# overview
df.head()

# %%
# Restrict to Tokyo
cond = (df["都道府県名"] == "東京都")
df2 = df.loc[cond].copy()
df2.head()

# %%


# %%
# Take log price
df["log_price"] = np.log(df["取引価格（総額）"])

# %%
# Much better
df["log_price"].hist(bins=50)
plt.title("Histogram of log price")
plt.show()

# %%
# summary statistics
df["log_price"].describe()

# %% [markdown]
# # Transaction date

# %%
# Check -> this is not ok
df["取引時期"].value_counts()

# %%
# Create a dataframe
df_time = pd.DataFrame(df["取引時期"].value_counts())
# Find how many unique quarters in the data
print(set(df_time.index.str[6]))
# make a dictionary out of it for further use
quarterly_dict = {'1': 1, '2': 4, '3': 7, '4': 10}

# %%


# %% [markdown]
# # Using jaconv

# %%
# Finally
transaction_date2date = {}
for i in range(len(df_time)):
    # the original str
    transaction_date = df_time.index[i]
    # year
    year_num = df_time.index[i][0:4]
    # quarterly to month
    month_num = quarterly_dict[df_time.index[i][6]]
    # transform to datetime format
    date = datetime.datetime.strptime(str(year_num) + str(month_num), "%Y%m")
    transaction_date2date.update({transaction_date: date})

# %%


# %%
# create longitude and latitude
df["date"] = df["取引時期"].map(transaction_date2date)

# %%
df["date"].head()

# %%


# %% [markdown]
# # Mean log price

# %%
# Know the difference: apply and transform
df["date_mean_log_price"] = df.groupby("date")["log_price"].transform(np.mean)

# apply
df_date = pd.DataFrame(df.groupby("date")["log_price"].apply(np.mean))

# %%
# Where did this seasonality come from?
plt.plot(df_date["log_price"])
plt.title("mean log price 2005-2023",size=16)
plt.xlabel("year",size=16)
plt.ylabel("mean log price",size=16)
plt.show()

# %%


# %% [markdown]
# # Area

# %%
# Check -> not okay
df["面積（㎡）"]

# %%
# create a dataframe
df_area = pd.DataFrame(df["面積（㎡）"].value_counts())

# square_meters converter
square_meters2area = {}
for i in range(len(df_area)):
    square_meters = df_area.index[i]
    area = re.sub("㎡以上", "", square_meters)
    area = re.sub("m&sup2;以上", "", area)
    area = re.sub(",", "", area)
    area = int(area)
    square_meters2area.update({square_meters: area})

# %%
# do not use replace, map is faster
df["area"] = df["面積（㎡）"].map(square_meters2area)

# %%
df["area"].hist(bins=50)
plt.title("Histogram of area")
plt.show()

# %%


# %% [markdown]
# # Type

# %%
type
# Check -> Seems Okay
df["種類"].value_counts()

# %%
# Create Dummies
# Note this could be only run once.
df = pd.get_dummies(df, columns=["種類"])

# %%


# %% [markdown]
# # Circumstances

# %%
# Check -> Interesting
df['取引の事情等'].value_counts()

# %%
# Create Dummies
df = pd.get_dummies(df, columns=['取引の事情等'])

# %%


# %% [markdown]
# # Create a feature dataframe and a target dataframe

# %%
df.columns

# %%


# %% [markdown]
# # Sample 10,000 records to reduce computation time

# %%
df_sample = df.sample(10000,random_state=123)

# %%


# %% [markdown]
# # Create Features and Target

# %%
def extract_target_feature(df_sample):
    # NOTE: I simply copy and pasted the ones I needed
    df_x = df_sample[[
        "area", "longitude", "latitude",
        '種類_宅地(土地)', '種類_宅地(土地と建物)',
        '取引の事情等_その他事情有り',
        '取引の事情等_他の権利・負担付き', 
        '取引の事情等_古屋付き・取壊し前提', '取引の事情等_瑕疵有りの可能性',
        '取引の事情等_私道を含む取引', '取引の事情等_調停・競売等', '取引の事情等_私道を含む取引、その他事情有り',
         '取引の事情等_調停・競売等、私道を含む取引',
        '取引の事情等_関係者間取引', '取引の事情等_関係者間取引、私道を含む取引', '取引の事情等_隣地の購入',
         '取引の事情等_隣地の購入、私道を含む取引', 
        '取引の事情等_隣地の購入、調停・競売等',
        '取引の事情等_隣地の購入、関係者間取引'
    ]]
    # Target
    df_y = df_sample[["log_price"]]
    return df_y,df_x

# %%
df_y,df_x = extract_target_feature(df_sample)

# %%


# %% [markdown]
# # Save and Load

# %%
# write to csv
df_x.to_csv("land_price_x.csv", index=False)
df_y.to_csv("land_price_y.csv", index=False)

# %%
# save as pickle
with open('df_x.pickle', 'wb') as f:
    pkl.dump(df_x, f)
with open('df_y.pickle', 'wb') as f:
    pkl.dump(df_y, f)

# %%
# load pickle
with open('df_x.pickle', 'rb') as f:
    df_x = pkl.load(f)
with open('df_y.pickle', 'rb') as f:
    df_y = pkl.load(f)

# %%


# %%


# %% [markdown]
# # Learning

# %%
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model

# %%


# %% [markdown]
# # Random Split

# %%
# train test split
x_train, x_test, y_train, y_test = train_test_split(df_x.values,df_y.values,
                                                    test_size=0.6,random_state=123)

# train dev test
#x_train, x_dev_test, y_train, y_dev_test = train_test_split(df_x.values,df_y.values,test_size=0.6)
# dev test split
#x_dev, x_test, y_dev, y_test = train_test_split(x_dev_test,y_dev_test,test_size=0.5)

# %%
y_train

# %%
y_train = np.reshape(y_train, [-1])
y_test  = np.reshape(y_test, [-1])
#y_dev   = np.reshape(y_dev, [-1])

x_train = x_train.astype(float)
y_train = y_train.astype(float)
x_test = x_test.astype(float)
y_test = y_test.astype(float)

# %%
x_train.shape

# %%
y_train.shape

# %%


# %% [markdown]
# # Linear Regression

# %%
# stats model
reg_linear = sm.OLS(y_train, x_train)
result = reg_linear.fit()
# prediction
y_test_linear = result.predict(x_test)

# %%
# Test Error
print(mean_squared_error(y_test_linear, y_test))

# %%
# OLS
result.summary()

# %%


# %%
plt.figure(figsize=(6,6))
plt.plot(y_test_linear,y_test,marker="o",linestyle="",alpha=0.6,color="black")
plt.title("linear regression",size=16)
plt.xlabel("prediction",size=16)
plt.ylabel("true",size=16)

# %%


# %%
# If you insist on using scikit here you go
#reg_linear = linear_model.LinearRegression()
#reg_linear.fit(x_train, y_train)
# prediction
#y_test_linear = reg_linear.predict(x_test)
#mean_squared_error(y_test_linear, y_test)

# %%


# %% [markdown]
# # Random forest

# %%
from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()

# %%
%%time
# Hyper Parameter Learning
if hyper_para_learn == 1:
    reg_rf_cv = GridSearchCV(reg_rf, {
        'max_depth': [2, 6, 10],
        'n_estimators': [100, 1000, 5000],
        'max_features': ['log2'],
        'n_jobs': [24]
    },
                             verbose=1)
    reg_rf_cv.fit(x_train, y_train)
    print(reg_rf_cv.best_params_)
    print(reg_rf_cv.best_score_)

# %%
# n_jobs should be adjusted to your computing environment
# in the server do not overuse computational resources
reg_rf = RandomForestRegressor(max_depth=10,
                               max_features='log2',
                               n_estimators=5000,
                               n_jobs=24)
reg_rf.fit(x_train, y_train)

# %%
# Prediction
y_test_rf = reg_rf.predict(x_test)

# %%
# Test error
mean_squared_error(y_test_rf, y_test)

# %%
plt.figure(figsize=(6,6))
plt.plot(y_test_rf,y_test,marker="o",linestyle="",alpha=0.6,color="black")
plt.title("random forest",size=16)
plt.xlabel("prediction",size=16)
plt.ylabel("true",size=16)

# %%
df_x.columns

# %%


# %%
# Importance
importances = pd.Series(reg_rf.feature_importances_, index=df_x.columns)
importances = importances.sort_values()
importances[-15:].plot(kind="barh")
plt.title("imporance (random forest)")
plt.show()

# %%


# %% [markdown]
# # Gradient Boosting

# %%
!pip install xgboost

# %%
import xgboost as xgb
#reg_xgb = xgb.XGBRegressor()

# %%
%%time
# Hyper Parameter Learning
if hyper_para_learn == 1:
    reg_xgb_cv = GridSearchCV(reg_xgb, {
        'max_depth': [2, 4, 6],
        'n_estimators': [50, 100, 500, 1000],
        'n_jobs': [24]
    },
                              verbose=1)
    reg_xgb_cv.fit(x_train, y_train)
    print(reg_xgb_cv.best_params_)
    print(reg_xgb_cv.best_score_)

# %%
# Set Parameters
reg_xgb = xgb.XGBRegressor(learning_rate=0.1,
                           n_estimators=500,
                           max_depth=4,
                           n_jobs=24)

# %%
# Fit
reg_xgb.fit(x_train, y_train)

# %%
# prediction
y_test_xgb = reg_xgb.predict(x_test)

# %%
# Test error
mean_squared_error(y_test_xgb, y_test)

# %%


# %% [markdown]
# # Scatter plot: real vs predicted value

# %%
plt.figure(figsize=(6,6))
plt.plot(y_test_xgb,y_test,marker="o",linestyle="",alpha=0.6,color="black")
plt.title("Gradient Boosting",size=16)
plt.xlabel("prediction",size=16)
plt.ylabel("true",size=16)

# %%
# Importance Measure
importances = pd.Series(reg_xgb.feature_importances_, index=df_x.columns)
importances = importances.sort_values()
importances[-15:].plot(kind="barh")
plt.title("imporance (xgboost)")
plt.show()

# %%


# %% [markdown]
# # Partial dependence

# %%
from sklearn.inspection import PartialDependenceDisplay
feature2index = dict()
for i in range(len(df_x.columns)):
    feature2index.update({df_x.columns[i]:i})
df_x.columns

# %%
feature = "area"
PartialDependenceDisplay.from_estimator(reg_xgb, x_train,
                                        [feature2index[feature]] )
plt.title(feature)

# %%
feature = '取引の事情等_調停・競売等'
PartialDependenceDisplay.from_estimator(reg_xgb, x_train,
                                        [feature2index[feature]] )
plt.title(feature)

# %%


# %%


# %% [markdown]
# # Assignment 1

# %% [markdown]
# # Restrict data

# %%
# Create Test data

# Restrict to dates before 2019-12-31 and Tokyo
# Sample 5000

# Restrict to dates before 2019-12-31
# Sample 5000


# %%


# %% [markdown]
# # Extract target and features

# %%
# Extract target and features df_test

# Extract target and features df_train_tokyo

# Extract target and features df_train_all

# %%


# %% [markdown]
# # Train Tokyo Only

# %%
# Fit
# Set Parameters

# %%
# Prediction

# %%
# Test error

# %%


# %% [markdown]
# # Scatter plot: real vs predicted value

# %%


# %% [markdown]
# # Train using all past

# %%
# Fit
# Set Parameters

# %%
# prediction

# %%
# Test error

# %%


# %% [markdown]
# # Scatter plot: real vs predicted value

# %%


# %%



