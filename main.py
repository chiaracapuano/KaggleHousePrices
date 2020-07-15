import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

pd.set_option("display.max_rows", 101)
train = pd.read_csv('/Users/chiara/PycharmProjects/KaggleHousePrices/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/Users/chiara/PycharmProjects/KaggleHousePrices/house-prices-advanced-regression-techniques/test.csv')

train = train.drop(columns = 'Id')
train['MSSubClass'] = train['MSSubClass'].astype(str)
print(train.dtypes)
