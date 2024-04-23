import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Sample data
data = {
    'chunks': ['apple', 'banana', 'orange', 'grape', 'watermelon', 'strawberry', 'pineapple', 'kiwi', 'mango', 'peach'],
    'embeddings': [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7],
        [0.4, 0.5, 0.6, 0.7, 0.8],
        [0.5, 0.6, 0.7, 0.8, 0.9],
        [0.6, 0.7, 0.8, 0.9, 1.0],
        [0.7, 0.8, 0.9, 1.0, 1.1],
        [0.8, 0.9, 1.0, 1.1, 1.2],
        [0.9, 1.0, 1.1, 1.2, 1.3],
        [1.0, 1.1, 1.2, 1.3, 1.4]
    ]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(df['embeddings'].tolist())

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='b', edgecolor='k', alpha=0.5)
for i, chunk in enumerate(df['chunks']):
    plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], chunk, fontsize=8)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Embeddings Visualization (2D)')
plt.grid(True)
plt.show()
