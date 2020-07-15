import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 101)
train = pd.read_csv('/Users/chiara/PycharmProjects/KaggleHousePrices/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/Users/chiara/PycharmProjects/KaggleHousePrices/house-prices-advanced-regression-techniques/test.csv')

train = train.drop(columns = ['Id', 'SalePrice'])
train['MSSubClass'] = train['MSSubClass'].astype(str)

numerical_cols = []
categorical_cols = []

for col in train.columns:
    if train[col].dtype == 'object':
        categorical_cols.append(col)
    else:
        numerical_cols.append(col)

#check which features need hot-encoding and which don't
trial_df = train[categorical_cols]#[['claps', 'publication', 'responses', 'reading_time' ]]
dfm = trial_df[[col for col in trial_df if trial_df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
if dfm.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({dfm.shape[1]}) is less than 2')
corr = dfm.corr()
plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
corrMat = plt.matshow(corr, fignum = 1)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.gca().xaxis.tick_bottom()
plt.colorbar(corrMat)
plt.show()




#categorical_cols_label_enc = []
# Log transform of the skewed numerical features to lessen impact of outliers
# Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
# As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
def transform_skewed(df, numerical_cols):
    for i in numerical_cols:
        skewness = df[i].apply(lambda x: skew(x))
        skewness = skewness[abs(skewness) > 0.5]
        skewed_features = skewness.index
        df[skewed_features] = np.log1p(df[skewed_features])



# Preprocessing for numerical data
numerical_transformer_NaN = SimpleImputer()
numerical_transformer_None = SimpleImputer(missing_values=None)


numerical_transformer = Pipeline(steps=[
    ('imputer_Nan', SimpleImputer()),
    ('imputer_None', SimpleImputer(missing_values=None)),
    ('de-skew', transform_skewed(train, numerical_cols)),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer_NaN, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])