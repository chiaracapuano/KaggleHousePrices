import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

pd.set_option("display.max_rows", 101)
train = pd.read_csv('/Users/chiara/PycharmProjects/KaggleHousePrices/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/Users/chiara/PycharmProjects/KaggleHousePrices/house-prices-advanced-regression-techniques/test.csv')

train['MSSubClass'] = train['MSSubClass'].astype(str)

X = train.drop(columns=['Id', 'SalePrice'])
y = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)


numerical_cols = []
categorical_cols = []

for col in train.columns:
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

corr_le = dfm_le.corr()
corr_le_keep = corr_le[abs(corr_le['SalePrice'])>0.4]
plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
corrMat = plt.matshow(corr_le, fignum = 1)
plt.xticks(range(len(corr_le.columns)), corr_le.columns, rotation=90)
plt.yticks(range(len(corr_le.columns)), corr_le.columns)
plt.gca().xaxis.tick_bottom()
plt.colorbar(corrMat)
#plt.show()





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
    #('de-skew', transform_skewed(train, numerical_cols)),
    ('scaler', StandardScaler())
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
    ('onehot', OrdinalEncoder())
])





# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer_NaN,numerical_cols),
        ('cat_ohe', categorical_transformer_ohe, cat_ohe),
        ('cat_le', categorical_transformer_le, cat_le)
    ])




model = GradientBoostingRegressor(random_state=0)
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_test)

# Evaluate the model
score = mean_absolute_error(y_test, preds)
print('MAE:', score)