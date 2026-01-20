# Model Guide

SCCL supports multiple classification models. This guide helps you choose the right model for your use case.

## Model Overview

| Model | Type | Training | Batch Correction | Speed | Best For |
|-------|------|----------|-----------------|-------|----------|
| **SCimilarity** | Foundation | None | ✓ | Fast | General-purpose, new datasets |
| **scVI** | Deep | Required | ✓ | Medium | Multi-study integration |
| **Random Forest** | ML | Required | ✗ | Fast | Baseline, interpretable |
| **SVM** | ML | Required | ✗ | Medium | Small datasets, clear boundaries |
| **Logistic Regression** | ML | Required | ✗ | Fast | Simple, interpretable |
| **KNN** | ML | None | ✗ | Fast | Reference mapping |

## Foundation Models

### SCimilarity

**What it is**: Pre-trained foundation model for cell type annotation.

**Pros**:
- No training needed
- Works well out-of-the-box
- Handles batch effects implicitly
- Fast inference
- Good generalization to new cell types

**Cons**:
- Requires scimilarity package
- Less interpretable
- May not capture dataset-specific patterns

**When to use**:
- First-time analysis of new datasets
- When you need quick results
- When you have batch effects
- When you don't have labeled training data

**Example**:
```python
from sccl import Pipeline

pipeline = Pipeline(model="scimilarity")
predictions = pipeline.predict(adata)
```

**Parameters**:
- `n_neighbors`: Neighbors for clustering (default: 15)
- `resolution`: Leiden clustering resolution (default: 1.0)

### scVI

**What it is**: Deep generative model for single-cell data.

**Pros**:
- Excellent batch correction
- Learns dataset-specific patterns
- Generates interpretable latent space
- Well-established method

**Cons**:
- Requires training (slower)
- Needs scvi-tools package
- Requires more data
- Sensitive to hyperparameters

**When to use**:
- Multi-study integration
- Strong batch effects
- Large datasets (>10k cells)
- When training time is acceptable

**Example**:
```python
pipeline = Pipeline(
    model="scvi",
    batch_key="study",
    model_params={
        'n_latent': 30,
        'max_epochs': 200
    }
)

predictions = pipeline.predict(adata)
```

**Parameters**:
- `n_latent`: Latent dimension (default: 30)
- `n_layers`: Network layers (default: 2)
- `max_epochs`: Training epochs (default: 400)

## Traditional Machine Learning

### Random Forest

**What it is**: Ensemble of decision trees.

**Pros**:
- Good baseline performance
- Handles non-linear relationships
- Feature importance available
- Robust to overfitting
- Fast training

**Cons**:
- Doesn't handle batch effects
- Requires labeled training data
- Large model size

**When to use**:
- Baseline comparison
- When interpretability matters
- Small to medium datasets
- Established cell type classification

**Example**:
```python
pipeline = Pipeline(
    model="random_forest",
    model_params={
        'n_estimators': 200,
        'max_depth': 10
    }
)

metrics = pipeline.evaluate(adata, target_column="cell_type")
```

**Parameters**:
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: None)

### SVM (Support Vector Machine)

**What it is**: Finds optimal decision boundaries between classes.

**Pros**:
- Works well with clear class separation
- Effective in high dimensions
- Various kernel options

**Cons**:
- Slow on large datasets
- Sensitive to scaling
- Doesn't handle batch effects
- Less interpretable with non-linear kernels

**When to use**:
- Small datasets (<10k cells)
- Well-separated cell types
- When you need strong theoretical guarantees

**Example**:
```python
pipeline = Pipeline(
    model="svm",
    model_params={
        'kernel': 'rbf',
        'C': 1.0
    }
)
```

**Parameters**:
- `kernel`: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
- `C`: Regularization strength (default: 1.0)

### Logistic Regression

**What it is**: Linear probabilistic classifier.

**Pros**:
- Very fast
- Highly interpretable
- Provides probability estimates
- Good for linear problems

**Cons**:
- Assumes linear boundaries
- May underfit complex data
- Doesn't handle batch effects

**When to use**:
- Need fast results
- Need probability estimates
- Linear cell type boundaries
- Interpretability is critical

**Example**:
```python
pipeline = Pipeline(
    model="logistic_regression",
    model_params={
        'penalty': 'l2',
        'C': 1.0
    }
)
```

**Parameters**:
- `penalty`: Regularization ('l1', 'l2', 'elasticnet', 'none')
- `C`: Inverse regularization strength (default: 1.0)

### KNN (K-Nearest Neighbors)

**What it is**: Classifies based on nearest neighbors.

**Pros**:
- No training needed
- Intuitive
- Works well for reference mapping
- Good for similar datasets

**Cons**:
- Slow on large datasets
- Sensitive to distance metric
- Doesn't handle batch effects
- Requires stored training data

**When to use**:
- Reference-based annotation
- Label transfer between similar studies
- Quick prototyping

**Example**:
```python
pipeline = Pipeline(
    model="knn",
    model_params={
        'n_neighbors': 10,
        'weights': 'distance'
    }
)
```

**Parameters**:
- `n_neighbors`: Number of neighbors (default: 5)
- `weights`: Weight function ('uniform', 'distance')
- `metric`: Distance metric (default: 'euclidean')

## Model Selection Guide

### Decision Tree

```
Do you have batch effects?
├─ Yes
│  ├─ Do you want to train a model?
│  │  ├─ Yes → scVI
│  │  └─ No → SCimilarity
│  └─ No → Continue below
└─ No
   ├─ Do you have labeled training data?
   │  ├─ Yes
   │  │  ├─ Large dataset (>50k cells) → Random Forest
   │  │  ├─ Small dataset (<10k cells) → SVM or Logistic Regression
   │  │  └─ Medium dataset → Random Forest or SVM
   │  └─ No
   │     ├─ SCimilarity (clustering)
   │     └─ KNN (if you have reference)
   └─ Do you need interpretability?
      ├─ Yes → Logistic Regression or Random Forest
      └─ No → Try all and compare
```

### By Use Case

**New Dataset Exploration**:
1. SCimilarity (first try)
2. Random Forest (if labels available)
3. Compare both

**Multi-Study Integration**:
1. scVI (best batch correction)
2. SCimilarity (faster alternative)

**Production/Deployment**:
1. SCimilarity (no retraining)
2. Logistic Regression (fast, simple)
3. Random Forest (good performance)

**Research/Publication**:
1. Compare multiple models
2. Use scVI or SCimilarity for main results
3. Include Random Forest as baseline

**Limited Computational Resources**:
1. Logistic Regression (fastest)
2. KNN (no training)
3. Random Forest (good balance)

## Performance Comparison

Based on typical single-cell datasets:

### Speed (Training + Prediction)
1. Logistic Regression: ~1 sec
2. KNN: ~2 sec (no training, slow prediction)
3. Random Forest: ~10 sec
4. SCimilarity: ~30 sec
5. SVM: ~1 min
6. scVI: ~5-20 min

### Typical Accuracy (depends on dataset)
1. SCimilarity: 0.85-0.95
2. scVI: 0.80-0.95
3. Random Forest: 0.75-0.90
4. SVM: 0.70-0.85
5. Logistic Regression: 0.65-0.85
6. KNN: 0.70-0.90

*Note: Actual performance varies greatly by dataset*

## Tips for Best Results

### General
- Always start with SCimilarity for quick assessment
- Compare multiple models
- Use appropriate test/train splits
- Check for data quality issues

### For Traditional ML
- Ensure proper preprocessing
- Use batch_key if available
- Try different hyperparameters
- Use cross-validation for small datasets

### For Foundation Models
- Ensure raw counts are available
- Let the model handle preprocessing
- Use for exploratory analysis
- Good for zero-shot transfer

## Combining Models

You can ensemble multiple models:

```python
from sccl import Pipeline
from scipy import stats

models = ['random_forest', 'svm', 'logistic_regression']
predictions_list = []

for model_name in models:
    pipeline = Pipeline(model=model_name)
    pred = pipeline.predict(adata, target_column='cell_type')
    predictions_list.append(pred)

# Majority voting
final_predictions = stats.mode(predictions_list, axis=0)[0]
```

## Next Steps

- See [User Guide](USER_GUIDE.md) for usage details
- Check [examples/](../examples/) for model-specific code
- Read [QUICKSTART.md](QUICKSTART.md) for getting started
