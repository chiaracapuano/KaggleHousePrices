import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy.stats import skew

pd.set_option("display.max_rows", 101)
train = pd.read_csv('/Users/chiara/PycharmProjects/KaggleHousePrices/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/Users/chiara/PycharmProjects/KaggleHousePrices/house-prices-advanced-regression-techniques/test.csv')

train = train.drop(columns = 'Id')
train['MSSubClass'] = train['MSSubClass'].astype(str)

numerical_cols = []
categorical_cols = []

for col in train.columns:
    if train[col].dtype == 'object':
        categorical_cols.append(col)
    else:
        numerical_cols.append(col)

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

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer_NaN, numerical_cols),
        ('num', numerical_transformer_None, numerical_cols),
        ('num', transform_skewed(train, numerical_cols)),
        ('cat', categorical_transformer, categorical_cols)
    ])