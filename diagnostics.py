import scanpy as sc
import numpy as np

adata = sc.read_h5ad("data/AML_scAtlas.h5ad")

print("Checking data structure:")
print(f"adata.X max: {adata.X.max()}")
print(f"adata.X min: {adata.X.min()}")
print(f"adata.X mean: {adata.X.mean()}")

if 'counts' in adata.layers:
    print(f"\nadata.layers['counts'] max: {adata.layers['counts'].max()}")
    print(f"adata.layers['counts'] min: {adata.layers['counts'].min()}")
    print(f"adata.layers['counts'] mean: {adata.layers['counts'].mean()}")
    
    # Check if they're the same
    if issparse(adata.X):
        same = np.allclose(adata.X.data, adata.layers['counts'].data)
    else:
        same = np.allclose(adata.X, adata.layers['counts'])
    print(f"\nAre X and layers['counts'] the same? {same}")

# Raw counts should be integers and have high max values (thousands)
# Normalized data should have max around 10-20 after log1p

