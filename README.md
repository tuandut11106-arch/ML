# 📚 Student Score Prediction

Dự án Machine Learning dùng để **dự đoán điểm Toán (`math score`) của học sinh** dựa trên các thông tin học tập và đặc điểm cá nhân trong bộ dữ liệu `StudentScore.xls`.

Project tập trung vào workflow:

* Phân tích dữ liệu & trực quan hóa
* Tiền xử lý dữ liệu bằng `ColumnTransformer` & `Pipeline`
* So sánh các mô hình baseline bằng Cross-Validation (`cross_val_score`)
* Lựa chọn `LinearRegression` và `LinearSVR`
* Tối ưu siêu tham số bằng `GridSearchCV`
* Đánh giá trên tập Test & hiển thị kết quả dự đoán

---

## 🛠️ Công nghệ & Thư viện sử dụng

* **Python 3.x**
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`

---

## 📂 Cấu trúc dự án

```text
PythonProject2/
├── dataset/          # Chứa tập dữ liệu (StudentScore.xls)
├── Image/            # Lưu các biểu đồ (Math Score.png)
├── main.py           # File huấn luyện và đánh giá mô hình
└── README.md         # Tài liệu hướng dẫn dự án

```

---

## 🔄 Workflow

```text
Dataset
   ↓
Phân tích dữ liệu & Trực quan hóa
   ↓
Chia Train / Test (80 / 20)
   ↓
Tiền xử lý dữ liệu (Imputer, Scaler, Encoder)
   ↓
So sánh Baseline Models (5-Fold CV)
   ↓
Chọn LinearRegression & LinearSVR
   ↓
GridSearchCV (6-Fold CV)
   ↓
Chọn model tốt nhất
   ↓
Đánh giá & Dự đoán trên tập Test

```

---

## 📈 Kết quả

Model tốt nhất sau `GridSearchCV`:

```text
LinearSVR

```

Best Parameters:

```text
C = 200
epsilon = 1.0
loss = squared_epsilon_insensitive

```

Best CV Score:

```text
R2 : 0.8658

```

Kết quả trên tập Test:

```text
MAE : ~4.20
MSE : ~29.00
R2  : ~0.866

```

---

## ▶️ Chạy project

```bash
python ex.py

```