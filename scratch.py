import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
pd.set_option("display.max_rows", 2000)
train_i = pd.read_csv('/Users/chiara/PycharmProjects/KaggleHousePrices/house-prices-advanced-regression-techniques/train.csv')
test_submission = pd.read_csv('/Users/chiara/PycharmProjects/KaggleHousePrices/house-prices-advanced-regression-techniques/test.csv')

train_i['MSSubClass'] = train_i['MSSubClass'].astype(str)








print('1')

numerical_cols = []
categorical_cols = []

for col in train_i.columns:
    if train_i[col].dtype == 'object':
        categorical_cols.append(col)
    else:
        numerical_cols.append(col)

numerical_cols.remove('SalePrice')
numerical_cols.remove('Id')


#check which features need hot-encoding and which don't
trial_df_le = pd.DataFrame()

le = LabelEncoder()
for col in categorical_cols:
    trial_df_le[col] = le.fit_transform(train_i[col].astype(str))

trial_df_le['SalePrice'] = train_i['SalePrice']



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
plt.show()




print('3')





# Preprocessing for numerical data
numerical_transformer_NaN = SimpleImputer()
numerical_transformer_None = SimpleImputer(missing_values=None)


numerical_transformer = Pipeline(steps=[
    ('imputer_Nan', SimpleImputer()),
    ('imputer_None', SimpleImputer(missing_values=None))
])


# Preprocessing for categorical data
cat_le = corr_le_keep.index.tolist()

cat_le.remove('SalePrice')
cat_ohe = list((Counter(categorical_cols)-Counter(cat_le)).elements())


for col in cat_le:
    train_i[col] = le.fit_transform(train_i[col].astype(str))

train_processed = pd.get_dummies(train_i,
                              columns=cat_ohe)



dummy_submission = pd.get_dummies(test_submission,
                              columns=cat_ohe)


for col in cat_le:
    dummy_submission[col] = le.fit_transform(test_submission[col].astype(str))

#dummy_submission.drop(columns = cat_le, inplace=True)


dummy_submission=dummy_submission.reindex(columns = train_processed.columns, fill_value=0)

train_inter, test_inter = train_test_split(
    train_processed, test_size=0.2)

train, test = train_inter.align(test_inter, join='outer', axis=1)
X_train = train.drop(columns=['Id', 'SalePrice'])
X_test = test.drop(columns=['Id', 'SalePrice'])

y_train = train['SalePrice']
y_test = test['SalePrice']


categorical_transformer_ohe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('OHE', OneHotEncoder())
])

categorical_transformer_le = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder())
])


print('4')
n_cat_ohe = X_train.columns
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        #('cat_ohe', categorical_transformer_ohe, n_cat_ohe),
        #('cat_le', categorical_transformer_le, cat_le),
        ('num', numerical_transformer_NaN, numerical_cols)


    ], remainder='passthrough')


# Bundle preprocessing and modeling code in a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('reduce_dim', SelectKBest(mutual_info_classif)),
                            ('clf', LGBMRegressor(random_state=0))
                             ])
print('5')


parameters = [{
        'reduce_dim__k': [200, 278],
        'clf__num_leaves':[20, 40],
        'clf__learning_rate': [0.1, 0.01],
        'clf__max_depth': [ 4, 2]}]



CV = GridSearchCV(pipeline, parameters, scoring = 'explained_variance', n_jobs=1)
CV.fit(X_train, y_train)
print(CV.best_params_)
best_params_dict = CV.best_params_
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('reduce_dim', SelectKBest(mutual_info_classif, k = best_params_dict["reduce_dim__k"])),
                            ('clf', LGBMRegressor(random_state=0, num_leaves = best_params_dict["clf__num_leaves"], learning_rate = best_params_dict["clf__learning_rate"], max_depth = best_params_dict["clf__max_depth"]

                                                              ))
                             ])

print('5')

# Preprocessing of training data, fit model
pipeline.fit(X_train, y_train)
# Preprocessing of validation data, get predictions
preds = pipeline.predict(X_test)
score = pipeline.score(X_test, y_test)
dummy_submission_to_pred = dummy_submission.drop(columns=['Id', 'SalePrice'])

preds = pipeline.predict(dummy_submission_to_pred)
print(preds)
df_submit = pd.DataFrame()
df_submit["Id"] = dummy_submission["Id"]

df_submit["SalePrice"]= preds
df_submit.to_csv('/Users/chiara/PycharmProjects/KaggleHousePrices/house-prices-advanced-regression-techniques/Chiara_best_model_submission.csv', index=False)
#TRIAL1: {'clf__learning_rate': 0.1, 'clf__max_depth': 4, 'clf__num_leaves': 30, 'reduce_dim__k': 278}
#Trial 2: {'clf__learning_rate': 0.1, 'clf__max_depth': 4, 'clf__num_leaves': 20, 'reduce_dim__k': 278}
