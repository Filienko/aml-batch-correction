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
    
# Raw counts should be integers and have high max values (thousands)
# Normalized data should have max around 10-20 after log1p

