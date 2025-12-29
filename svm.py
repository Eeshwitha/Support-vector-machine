import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create non-linearly separable ring data
X, y = make_circles(
    n_samples=300,
    factor=0.4,     # inner circle size
    noise=0.05,
    random_state=42
)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr")
plt.title("Ring-shaped Binary Classification Data")
plt.show()

# Pipeline: scaling + SVM
rbf_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

rbf_svm.fit(X, y)

print("Number of support vectors:",
      rbf_svm.named_steps["svm"].n_support_)

svm_c1 = SVC(kernel="linear", C=1.0)
svm_c10 = SVC(kernel="linear", C=10.0)

svm_c1.fit(X, y)
svm_c10.fit(X, y)

print("Support vectors (C=1):", svm_c1.n_support_)
print("Support vectors (C=10):", svm_c10.n_support_)


# Original training
svm = SVC(kernel="linear", C=1.0)
svm.fit(X, y)
original_sv = svm.support_vectors_.shape[0]

# Remove points far from boundary (simulate duplicates removal)
X_reduced = X[:50]
y_reduced = y[:50]

svm_retrained = SVC(kernel="linear", C=1.0)
svm_retrained.fit(X_reduced, y_reduced)
new_sv = svm_retrained.support_vectors_.shape[0]

print("Support vectors before:", original_sv)
print("Support vectors after removing duplicates:", new_sv)


# Artificial unscaled data
X_unscaled = np.column_stack((
    np.random.rand(100),          # 0–1
    np.random.rand(100) * 10000    # 0–10000
))
y = np.random.choice([0, 1], 100)

# Without scaling
svm_no_scale = SVC(kernel="linear")
svm_no_scale.fit(X_unscaled, y)

# With scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unscaled)

svm_scaled = SVC(kernel="linear")
svm_scaled.fit(X_scaled, y)

print("Coefficients without scaling:", svm_no_scale.coef_)
print("Coefficients with scaling:", svm_scaled.coef_)


#import numpy as np

w = np.array([0.5, -1.2])
b = -0.3
x = np.array([4, 3])

decision_value = np.dot(w, x) + b
prediction = "Positive" if decision_value > 0 else "Negative"

print("Decision function value:", decision_value)
print("Predicted class:", prediction)
