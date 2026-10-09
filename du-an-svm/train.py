import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Tải dữ liệu ung thư vú Wisconsin
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

# 2. Chia tập train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Pipeline: Chuẩn hóa + Mô hình SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])

# 4. Tìm tham số tốt nhất
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['rbf', 'linear']
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 5. Lưu mô hình thành file
best_model = grid_search.best_estimator_
payload = {
    'model': best_model,
    'feature_names': list(X.columns),
    'target_names': list(data.target_names)
}
joblib.dump(payload, 'svm_breast_cancer_model.joblib')

print("Huấn luyện thành công! Đã tạo file: svm_breast_cancer_model.joblib")