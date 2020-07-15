import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statistics

pd.set_option("display.max_rows", 2000)
train = pd.read_csv('/Users/chiara/PycharmProjects/KaggleHousePrices/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/Users/chiara/PycharmProjects/KaggleHousePrices/house-prices-advanced-regression-techniques/test.csv')

train['MSSubClass'] = train['MSSubClass'].astype(str)
print(train['SalePrice'])

X = train.drop(columns=['Id', 'SalePrice'])
sc_Y = MinMaxScaler()
y = sc_Y.fit_transform(train[['SalePrice']])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)

print('1')

numerical_cols = []
categorical_cols = []

for col in X.columns:
    if train[col].dtype == 'object':
        categorical_cols.append(col)
    else:
        numerical_cols.append(col)

#check which features need hot-encoding and which don't
trial_df_le = pd.DataFrame()
trial_df_enc = pd.DataFrame()

le = LabelEncoder()
for col in categorical_cols:
    trial_df_le[col] = le.fit_transform(train[col].astype(str))


trial_df_le['SalePrice'] = train['SalePrice']
dfm_le = trial_df_le[[col for col in trial_df_le if trial_df_le[col].nunique() > 1]] # keep columns where there are more than 1 unique values
print('2')

corr_le = dfm_le.corr()
corr_le_keep = corr_le[abs(corr_le['SalePrice'])>0.4]
plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
corrMat = plt.matshow(corr_le, fignum = 1)
plt.xticks(range(len(corr_le.columns)), corr_le.columns, rotation=90)
plt.yticks(range(len(corr_le.columns)), corr_le.columns)
plt.gca().xaxis.tick_bottom()
plt.colorbar(corrMat)
#plt.show()




print('3')





# Preprocessing for numerical data
numerical_transformer_NaN = SimpleImputer()
numerical_transformer_None = SimpleImputer(missing_values=None)


numerical_transformer = Pipeline(steps=[
    ('imputer_Nan', SimpleImputer()),
    ('imputer_None', SimpleImputer(missing_values=None)),
    ('scaler', MinMaxScaler())
])



# Preprocessing for categorical data
cat_le = corr_le_keep.index.tolist()
cat_le.remove('SalePrice')
cat_ohe = list((Counter(categorical_cols)-Counter(cat_le)).elements())

categorical_transformer_ohe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

categorical_transformer_le = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder())
])


print('4')


# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat_ohe', categorical_transformer_ohe, cat_ohe),
        ('cat_le', categorical_transformer_le, cat_le),
        ('num', numerical_transformer_NaN, numerical_cols)

    ])




model = GradientBoostingRegressor(random_state=0)
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
print('5')

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)
# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_test)
print(len(X_test))
print(type(preds))

# Evaluate the model
score = mean_squared_error(y_test, preds)
print('MSE:', score)
yhat = sc_Y.inverse_transform(preds.reshape(-1, 1))
y_test_transf = sc_Y.inverse_transform(y_test)

a = list(zip(yhat, y_test_transf))
print(a)