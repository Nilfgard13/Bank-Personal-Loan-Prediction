from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle

from fastapi.middleware.cors import CORSMiddleware

# =====================================================
# 1. Definisi Kelas KernelSVM (copy dari Jupyter)
# =====================================================
class KernelSVM:
    def __init__(
        self,
        kernel='linear',
        learning_rate=0.001,
        epochs=500,
        C=1.0,
        degree=3,
        gamma=1.0,
        coef0=1.0
    ):
        """
        SVM kernel (manual, pakai formulasi dual).
        """
        self.kernel = kernel
        self.lr = learning_rate
        self.epochs = epochs
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

        self.alphas = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.losses = []

    def _kernel(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'poly':
            K = X1 @ X2.T
            K = self.gamma * K + self.coef0
            return np.power(K, self.degree)
        elif self.kernel == 'rbf':
            X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
            X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True).T
            sq_dists = X1_sq + X2_sq - 2 * (X1 @ X2.T)
            return np.exp(-self.gamma * sq_dists)
        else:
            raise ValueError(f"Kernel tidak dikenal: {self.kernel}")

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        # label {0,1} -> {-1, +1}
        y = np.where(y == 0, -1.0, 1.0)

        n_samples = X.shape[0]
        self.X_train = X
        self.y_train = y

        K = self._kernel(X, X)

        self.alphas = np.zeros(n_samples)
        self.losses = []

        for epoch in range(self.epochs):
            v = self.alphas * y
            f = K @ v
            g = y * f - 1.0

            self.alphas -= self.lr * g
            self.alphas = np.clip(self.alphas, 0.0, self.C)

            v = self.alphas * y
            obj = 0.5 * np.dot(v, K @ v) - np.sum(self.alphas)
            self.losses.append(obj)

        v = self.alphas * y
        f = K @ v

        sv_mask = (self.alphas > 1e-4) & (self.alphas < self.C - 1e-4)
        if np.any(sv_mask):
            b_vals = y[sv_mask] - f[sv_mask]
        else:
            nz_mask = self.alphas > 1e-4
            if np.any(nz_mask):
                b_vals = y[nz_mask] - f[nz_mask]
            else:
                b_vals = np.array([0.0])

        self.b = float(np.mean(b_vals))
        return self

    def decision_function(self, X):
        X = np.array(X, dtype=float)
        K_test = self._kernel(X, self.X_train)
        v = self.alphas * self.y_train
        return K_test @ v + self.b

    def predict(self, X):
        f = self.decision_function(X)
        return np.where(f >= 0, 1, 0)


# =====================================================
# 2. Custom Unpickler untuk mapping __main__.KernelSVM
# =====================================================
class KernelSVMUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Model lama tersimpan sebagai __main__.KernelSVM
        if module == "__main__" and name == "KernelSVM":
            return KernelSVM
        return super().find_class(module, name)


# =====================================================
# 3. Load artifacts dari file .pkl pakai Custom Unpickler
# =====================================================
with open("svm_rbf_bank_loan_model.pkl", "rb") as f:
    artifacts = KernelSVMUnpickler(f).load()

feature_cols = artifacts["feature_cols"]     # urutan kolom fitur
scaler_after = artifacts["scaler_after"]     # StandardScaler setelah SMOTE
pca_after    = artifacts["pca_after"]        # PCA(n_components=3)
svm_rbf      = artifacts["svm_rbf"]          # instance KernelSVM (rbf)


# =====================================================
# 4. Skema input untuk FastAPI
# =====================================================
class LoanFeatures(BaseModel):
    Age: int
    Experience: int
    Income: float
    Family: int
    CCAvg: float
    Education: int
    Mortgage: float
    Securities_Account: int
    CD_Account: int
    Online: int
    CreditCard: int


# =====================================================
# 5. Inisialisasi FastAPI
# =====================================================
app = FastAPI(
    title="Bank Personal Loan – Kernel SVM API",
    description=(
        "API untuk memprediksi penerimaan personal loan menggunakan "
        "pipeline: StandardScaler -> PCA -> KernelSVM (RBF)"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Personal Loan SVM API is running"}


# =====================================================
# 6. Endpoint Prediksi
# =====================================================
@app.post("/predict")
@app.post("/predict")
def predict_loan(data: LoanFeatures):
    """
    Terima 1 sampel nasabah, kembalikan prediksi:
    - 0 = tidak menerima loan
    - 1 = menerima loan
    """
    # 1. Ambil data sebagai dict dari Pydantic
    d = data.dict()

    # 2. Mapping: nama kolom di dataset (feature_cols)
    #    -> nama field yang datang dari client (JSON)
    name_mapping = {
        "Securities Account": "Securities_Account",
        "CD Account": "CD_Account",
        # kolom lain (Age, Experience, dst) namanya sudah sama
    }

    # 3. Susun satu baris data dengan KEY = nama kolom di feature_cols
    row = {}
    for col in feature_cols:
        if col in name_mapping:
            field_name = name_mapping[col]   # ambil dari JSON dengan nama underscore
        else:
            field_name = col                 # nama sudah sama

        row[col] = d[field_name]

    # 4. Buat DataFrame dengan kolom PERSIS seperti feature_cols
    df_input = pd.DataFrame([row])
    X = df_input[feature_cols]

    # 5. Pipeline yang sama seperti di notebook
    X_scaled = scaler_after.transform(X)
    X_pca    = pca_after.transform(X_scaled)
    X_svm    = X_pca[:, :2]  # karena training: X_svm = X_after_pca[:, :2]

    # 6. Prediksi
    y_pred = int(svm_rbf.predict(X_svm)[0])
    score  = float(svm_rbf.decision_function(X_svm)[0])

    return {
        "prediction": y_pred,
        "decision_score": score,
        "feature_order": feature_cols,
    }
