import warnings
warnings.filterwarnings("ignore")


from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # We'll use only sepal length and sepal width
y = iris.target
target_names = iris.target_names

# Original data
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
for i, target_name in enumerate(target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=target_name)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Original Data')
plt.legend()

# Z-score scaled data
scaler_zscore = StandardScaler()
X_zscore = scaler_zscore.fit_transform(X)
plt.subplot(1, 3, 2)
for i, target_name in enumerate(target_names):
    plt.scatter(X_zscore[y == i, 0], X_zscore[y == i, 1], label=target_name)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Z-Score Scaled Data')

# Min-Max scaled data
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)
plt.subplot(1, 3, 3)
for i, target_name in enumerate(target_names):
    plt.scatter(X_minmax[y == i, 0], X_minmax[y == i, 1], label=target_name)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Min-Max Scaled Data')

# Show plots
plt.tight_layout()
plt.legend()
plt.show()

