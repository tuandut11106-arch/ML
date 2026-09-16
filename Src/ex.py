# =========================
# 1. IMPORT THƯ VIỆN
# =========================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sn
from lazypredict.Supervised import LazyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# =========================
# 2. ĐỌC VÀ XEM DỮ LIỆU
# =========================
df = pd.read_csv("../dataset/StudentScore.xls")
Target = "math score"
x = df.drop(Target, axis=1)
y = df[Target]

print("\n========== THÔNG TIN DỮ LIỆU ==========")
print("Số mẫu:", df.shape[0])
print("Số feature:", x.shape[1])

sn.histplot(df[Target])
plt.title('Math Score')
plt.savefig(
    r"C:\Users\ADMIN\PycharmProjects\PythonProject2\Image\Math Score.png"
)

print("\n========== MA TRẬN TƯƠNG QUAN ==========")
print(df.select_dtypes(include="number").corr())
print("==> Các feature có tương quan cao nên có khả năng các mô hình tuyến tính sẽ có hiệu quả tốt")

# =========================
# 3. CHIA DỮ LIỆU TRAIN VÀ TEST
# =========================
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\n========== CHIA DỮ LIỆU ==========")
print("Train:", x_train.shape)
print("Test :", x_test.shape)

# =========================
# 4. TIỀN XỬ LÍ DỮ LIỆU
# =========================
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

#print(x['parental level of education'].unique())
education_values = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"]
gender_values = ['male', 'female']
lunch_values = x_train['lunch'].unique()
test_values = x_train['test preparation course'].unique()
# result = num_transformer.fit_transform(x_train)
ord_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values]))
])

nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('encoder', OneHotEncoder(sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, ['reading score', 'writing score']),
    ('ordinal', ord_transformer, ['parental level of education', 'gender', 'lunch', 'test preparation course']),
    ('nom', nom_transformer, ['race/ethnicity'])
])

# =========================
# 5. SO SÁNH CÁC MÔ HÌNH BẰNG CROSS-VALIDATION
# =========================
print("\n========== DỰ ĐOÁN THỬ BẰNG CÁC MÔ HÌNH BASELINE ==========")

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# Danh sách các mô hình muốn so sánh
baseline_models = {
    'Linear Regression': LinearRegression(),
    'Linear SVR': LinearSVR(random_state=42, max_iter=10000),
    'Ridge Regression': Ridge(),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Chạy thử từng mô hình với 5-Fold Cross Validation
for name, model_obj in baseline_models.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model_obj)
    ])

    # Tính điểm R2 trung bình trên tập train
    scores = cross_val_score(pipe, x_train, y_train, cv=5, scoring='r2')
    print(f"{name:<20} | R2 Score (CV): {scores.mean():.4f}")

print('\n==> Linear Regression và Linear SVR cho kết quả tốt và tốc độ xử lý nhanh, '
      'chọn 2 mô hình này để tiếp tục tối ưu.')

# =========================
# 6. TỐI ƯU HYPERPARAMETER BẰNG GRIDSEARCHCV
# =========================
print("\n========== TỐI ƯU BẰNG GRIDSEARCHCV ==========")

# Danh sách (Tên mô hình, Mô hình, Tham số)
search_space = [
    ("LinearRegression", LinearRegression(), {'regressor__fit_intercept': [True, False]}),
    ("LinearSVR", LinearSVR(random_state=42, max_iter=50000), {
        'regressor__C': [1, 10, 100, 200],
        'regressor__epsilon': [0.0, 0.1, 0.5, 1.0],
        'regressor__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
    })
]

best_score = -1

for name, regressor, params in search_space:
    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', regressor)])
    grid = GridSearchCV(pipe, param_grid=params, scoring='r2', cv=6, n_jobs=-1).fit(x_train, y_train)

    print(f"{name:<18} | Best R2: {grid.best_score_:.4f} | Best Params: {grid.best_params_}")

    # Lưu lại model có R2 cao nhất để chạy cho Mục 7
    if grid.best_score_ > best_score:
        best_score, model = grid.best_score_, grid.best_estimator_

print(f"\n==> MODEL TỐT NHẤT: {model.named_steps['regressor'].__class__.__name__} (R2 = {best_score:.4f})")

# =========================
# 7. DỰ ĐOÁN VÀ ĐÁNH GIÁ MODEL
# =========================
y_predict = model.predict(x_test)

print("\n========== KẾT QUẢ TRÊN TẬP TEST ==========")
print(f"MAE : {mean_absolute_error(y_test, y_predict):.3f}")
print(f"MSE : {mean_squared_error(y_test, y_predict):.3f}")
print(f"R2  : {r2_score(y_test, y_predict):.3f}")

# IN THỬ 10 KẾT QUẢ DỰ ĐOÁN ĐẦU TIÊN ĐỂ XEM OUTPUT
results_df = pd.DataFrame({'Điểm thật': y_test.values[:10], 'Điểm dự đoán': y_predict[:10].round(1)})
print("\nSo sánh 10 mẫu đầu tiên:")
print(results_df)