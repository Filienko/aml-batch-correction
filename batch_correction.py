from pathlib import Path
import warnings
import os
import sys
import json
import copy
import gc
import time
import shutil
import traceback
from typing import List, Tuple, Dict, Union, Optional
import logging
import pickle

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.sparse import issparse
import scib_metrics

import scanpy as sc
import anndata
from anndata import AnnData
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from collections import Counter
import torch
import wandb

sys.path.insert(0, "../")

# Plotting settings
plt.style.context('default')
sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.simplefilter("ignore", ResourceWarning)
warnings.filterwarnings('ignore')

# === Config ===
dataset_name = "scAML"
save_dir = Path("processed/")
save_dir.mkdir(parents=True, exist_ok=True)

def load_dataset(filename):
    """Load dataset with memory optimization"""
    adata = sc.read_h5ad(filename)

    # Convert to sparse if not already
    if not scipy.sparse.issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    # Optimize data types
    adata.X.data = adata.X.data.astype(np.float32)  # 32-bit instead of 64-bit

    # Clean up categorical columns (removes unused categories)
    for col in adata.obs.select_dtypes(include=['category']):
        adata.obs[col] = adata.obs[col].cat.remove_unused_categories()

    return adata

# adata_full = load_dataset("data/AML_scAtlas.h5ad")
adata_scVI = load_dataset("data/AML_scAtlas_X_scVI.h5ad")

# print("Loaded ", adata_full)
print("Loaded ", adata_scVI)
