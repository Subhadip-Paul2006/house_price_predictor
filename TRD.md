<![CDATA[<div align="center">

# 🔧 Technical Requirements Document (TRD)

### House Price Predictor — v1.0

</div>

---

## 1. System Architecture

```mermaid
flowchart TB
    subgraph DataLayer ["📂 Data Layer"]
        CSV[("house_data.csv<br/>1000 rows")]
        Gen["generate_data.py"]
        Gen -->|creates| CSV
    end

    subgraph ProcessingLayer ["🧹 Processing Layer"]
        Pre["preprocess.py<br/>HouseData + Preprocessor"]
        CSV --> Pre
        Pre -->|clean + encode + scale| Encoded["Encoded Features"]
    end

    subgraph TrainingLayer ["🤖 Training Layer"]
        Train["train.py<br/>ModelTrainer"]
        Eval["evaluate.py<br/>Evaluator"]
        Encoded --> Train
        Train --> Eval
        Eval --> Artifacts[("models/<br/>model.joblib<br/>metrics.json")]
    end

    subgraph InferenceLayer ["🔮 Inference Layer"]
        Model["model.py<br/>ArtifactStore + Predictor"]
        Artifacts --> Model
    end

    subgraph UILayer ["🌐 UI Layer"]
        App["app.py<br/>Streamlit"]
        Model --> App
        App --> Predict["🔮 Predict"]
        App --> Compare["⚖️ Compare"]
        App --> Charts["📊 Charts"]
        App --> PDF["📄 PDF"]
    end

    style DataLayer fill:#E8F5E9
    style ProcessingLayer fill:#E3F2FD
    style TrainingLayer fill:#FFF3E0
    style InferenceLayer fill:#F3E5F5
    style UILayer fill:#FFEBEE
```

---

## 2. Module Dependency Graph

```mermaid
flowchart LR
    app["app.py"] --> model["model.py"]
    train["train.py"] --> model
    train --> preprocess["preprocess.py"]
    train --> evaluate["evaluate.py"]
    app --> preprocess

    model -.->|joblib| artifacts[("models/")]
    preprocess -.->|pandas| data[("data/")]

    style app fill:#FF4B4B,color:#fff
    style model fill:#2196F3,color:#fff
    style train fill:#FF9800,color:#fff
    style preprocess fill:#4CAF50,color:#fff
    style evaluate fill:#9C27B0,color:#fff
```

---

## 3. Data Schema

### 3.1 Raw Dataset (`house_data.csv`)

| Column | Type | Range | Nullable | Description |
|--------|------|-------|----------|-------------|
| Area | float | 500–5000 | 2% NaN | Built-up area (sq ft) |
| Bedrooms | float | 1–6 | 1% NaN | Number of bedrooms |
| Bathrooms | int | 1–5 | No | Number of bathrooms |
| Age | int | 0–50 | No | Property age (years) |
| Location | str | 4 categories | 1.5% NaN | Downtown/Urban/Suburban/Rural |
| Price | float | ≥ 5 L | No | Target variable (Lakhs ₹) |

### 3.2 Encoding Pipeline

```mermaid
flowchart LR
    subgraph Numeric ["Numeric: Area, Bedrooms, Bathrooms, Age"]
        N1["SimpleImputer<br/>strategy=median"] --> N2["StandardScaler"]
    end

    subgraph Categorical ["Categorical: Location"]
        C1["OneHotEncoder<br/>handle_unknown=ignore"]
    end

    N2 --> CT["ColumnTransformer"]
    C1 --> CT
    CT --> Output["Encoded Feature Matrix<br/>8 columns"]

    style Numeric fill:#E3F2FD
    style Categorical fill:#FFF3E0
```

---

## 4. ML Model Specifications

### 4.1 Algorithms

| Parameter | Linear Regression | Random Forest |
|-----------|:-----------------:|:-------------:|
| **Class** | `LinearRegression()` | `RandomForestRegressor` |
| n_estimators | — | 200 |
| max_depth | — | 20 |
| random_state | — | 42 |
| Train/Test | 80/20 | 80/20 |
| CV Folds | 5 | 5 |

### 4.2 Metrics

| Metric | Linear Regression | Random Forest |
|--------|:-----------------:|:-------------:|
| R² | 0.9018 | **0.9313** |
| MAE | 19.38 L | **15.11 L** |
| RMSE | 24.82 L | **20.77 L** |
| CV R² | 0.8782 | **0.8816** |
| Residual Std | 24.87 L | **20.81 L** |

---

## 5. Training Pipeline

```mermaid
flowchart LR
    S1["1. Load CSV"] --> S2["2. Clean Data"]
    S2 --> S3["3. Engineer Features"]
    S3 --> S4["4. Split 80/20"]
    S4 --> S5["5. Scale + Encode"]
    S5 --> S6["6. Train Model"]
    S6 --> S7["7. Cross-Validate"]
    S7 --> S8["8. Evaluate Test"]
    S8 --> S9["9. Save Artifacts"]

    style S1 fill:#4CAF50,color:#fff
    style S6 fill:#2196F3,color:#fff
    style S9 fill:#FF9800,color:#fff
```

---

## 6. Artifact Storage

```
models/
├── linear_regression/
│   └── 20260624_145024/
│       ├── model.joblib      # LinearRegression + Preprocessor bundle
│       └── metrics.json      # {mae, mse, rmse, r2, residual_std, ...}
└── random_forest/
    └── 20260624_145030/
        ├── model.joblib      # RandomForest + Preprocessor bundle (14.6 MB)
        └── metrics.json
```

---

## 7. API Reference

### `Predictor.predict(features: dict) → float`

Returns the predicted price in Lakhs ₹.

### `Predictor.confidence(features: dict) → Tuple[float, float, float]`

Returns `(price, lower_bound, upper_bound)` with 95% CI.

### `ArtifactStore.save(model, preprocessor, metrics, algorithm_name) → str`

Persists model bundle and metrics to disk. Returns artifact directory path.

### `ArtifactStore.load(algorithm_name, run_id=None) → Tuple`

Loads the latest or specified run. Returns `(model, preprocessor, metrics)`.

---

## 8. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥ 2.0 | Data handling |
| numpy | ≥ 1.24 | Numerical ops |
| scikit-learn | ≥ 1.4 | ML models + preprocessing |
| joblib | ≥ 1.3 | Model persistence |
| matplotlib | ≥ 3.7 | Charting |
| seaborn | ≥ 0.13 | Statistical plots |
| streamlit | ≥ 1.30 | Web UI |
| fpdf | ≥ 1.7.2 | PDF generation |

---

<div align="center">

_TRD v1.0 — House Price Predictor_

</div>
]]>
