import warnings
warnings.filterwarnings("ignore")


from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='FlowerType')
# print(X.head())
pca_iris = PCA(n_components=3).fit(iris.data)
print(pca_iris)
print(pca_iris.explained_variance_ratio_)
print(pca_iris.components_)
print(pca_iris.transform(iris.data))

print("Explained variance ratio:", pca_iris.explained_variance_ratio_)

transformed_data = pca_iris.transform(iris.data)

# Plot
colors = ['r', 'g', 'b']
markers = ['o', 's', 'D']

fig, ax = plt.subplots()

for i, flower_type in enumerate(iris.target_names):
    ax.scatter(transformed_data[y == i, 0], transformed_data[y == i, 1],
               c=colors[i],
               marker=markers[i],
               label=flower_type)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA of Iris Dataset')

plt.legend()
plt.show()
