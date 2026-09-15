import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sn

data = pd.read_csv("../dataset/StudentScore.xls")
target = "math score"
# sn.histplot(data["math score"])
# plt.title("Math score distribution")
# plt.savefig("MathDistribution.png")
x = data.drop(target, axis=1)
y = data[target]
# print(x[0])
# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
# imputer = SimpleImputer(strategy="mean")
# x["reading score"] = imputer.fit_transform(x[["reading score"]])
# scaler = StandardScaler()
# x["reading score"] = imputer.fit_transform(x[["reading score"]])
# print(x["reading score"])
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
# print(x["parental level of education"].unique())
education_value = ["some high school", "high school", "associate's degree", "master's degree", "some college", "bachelor's degree"]
gender_values = ["male", "female"]
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("encoder", OrdinalEncoder(categories=[education_value, gender_values, lunch_values, test_values]))
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("encoder", OneHotEncoder(sparse_output=False))
])
# result = nom_transformer.fit_transform(x_train[["race/ethnicity"]])
# for i,j in zip(x["race/ethnicity"], result):
#     print("Before {} After {}".format(i,j))
preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("ordinal_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nominal_features", nom_transformer, ["race/ethnicity"]),
])

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", SVR()),
])
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
for i, j in zip(y_test, y_predict):
    print("Actual:", i, "Predicted:", j)