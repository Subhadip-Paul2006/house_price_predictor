# 🏛️ ARCHITECTURE.md — House Price Prediction App

> Technical architecture reference: system design, C4 model, class structure, Wardley Map, and design decisions.

---

## 📑 Table of Contents

1. [Design Principles](#-design-principles)
2. [High-Level Architecture](#-high-level-architecture)
3. [C4 Model](#-c4-model)
   - [Level 1 — System Context](#level-1--system-context)
   - [Level 2 — Container](#level-2--container)
   - [Level 3 — Component](#level-3--component)
4. [Class Diagram](#-class-diagram)
5. [Wardley Map](#-wardley-map)
6. [Data Flow](#-data-flow)
7. [Components & Responsibilities](#-components--responsibilities)
8. [Design Decisions (ADR-style)]#️-design-decisions-adr-style)
9. [Non-Functional Requirements](#-non-functional-requirements)
10. [Risk Register](#-risk-register)

---

## 🎯 Design Principles

| Principle | How it's applied |
|-----------|------------------|
| **Separation of concerns** | Data, modeling, and UI live in distinct modules; `app.py` never imports sklearn internals directly, only `model.predict`. |
| **Reproducibility** | All randomness is seeded (`random_state=42`); model artifacts are versioned by training date. |
| **Idempotent pipeline** | Re-running `train.py` from raw CSV yields the same model artifact. |
| **Thin UI, fat model** | Streamlit is a presentation layer; all business logic is in `preprocess.py` / `model.py`. |
| **Fail loud, fail early** | Invalid inputs raise typed exceptions surfaced as user-friendly Streamlit errors. |

---

## 🏗️ High-Level Architecture

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        USER([👤 End User])
    end

    subgraph Presentation["Presentation Layer"]
        UI["Streamlit App<br/>app.py"]
    end

    subgraph Logic["Business Logic Layer"]
        PRE["Preprocessor<br/>preprocess.py"]
        MODEL["Predictor<br/>model.py"]
        METRICS["Evaluator<br/>evaluate.py"]
    end

    subgraph Data["Data Layer"]
        CSV[("house_data.csv")]
        ART[("models/*.pkl")]
    end

    USER -->|HTTP via browser| UI
    UI -->|features dict| PRE
    PRE -->|scaled X| MODEL
    MODEL -->|price estimate| UI
    CSV -->|load| PRE
    CSV -->|train/test split| METRICS
    METRICS -->|MAE/RMSE/R2| ART
    ART <-->|load / save| MODEL

    classDef client fill:#ffd166,color:#000,stroke:#f4a261,stroke-width:2px;
    classDef ui fill:#ef476f,color:#fff,stroke:#d62828;
    classDef logic fill:#06d6a0,color:#000,stroke:#1b9e8a,stroke-width:2px;
    classDef data fill:#118ab2,color:#fff,stroke:#073b4c;

    class USER client;
    class UI ui;
    class PRE,MODEL,METRICS logic;
    class CSV,ART data;
```

---

## 🧱 C4 Model

### Level 1 — System Context

```mermaid
C4Context
    title House Price Prediction — System Context

    Person(user, "End User", "Wants an instant house-price estimate")
    System_Boundary(hpp, "House Price Prediction") {
        Container(app, "Streamlit Web App", "Python / Streamlit", "Captures inputs, returns predictions")
    }
    System_Ext(data_src, "Public Dataset", "Kaggle / web-scraped real-estate CSVs")
    System_Ext(github, "GitHub", "Source control & CI")

    Rel(user, app, "Uses", "HTTPS / browser")
    Rel(app, data_src, "Sources training data", "manual download")
    Rel(github, app, "Deploys via", "Git pull / CI")
```

### Level 2 — Container

```mermaid
C4Context
    title House Price Prediction — Container View

    Person(user, "End User", "Real-estate buyer / enthusiast")
    System_Boundary(hpp, "House Price Prediction") {
        Container(web, "Streamlit UI", "Python · Streamlit", "Form inputs, charts, report download")
        Container(pipeline, "Training Pipeline", "Python · scikit-learn · ZenML", "Ingest, clean, train, evaluate")
        ContainerDb(artifacts, "Model Store", "filesystem · .pkl/.joblib", "Serialized models + metrics")
        ContainerDb(data, "Data Store", "filesystem · CSV/Parquet", "Raw & processed datasets")
    }
    System_Ext(datasource, "Public Datasets", "Kaggle / portals")

    Rel(user, web, "Opens in browser", "HTTP")
    Rel(web, artifacts, "Loads model for inference", "file read")
    Rel(pipeline, data, "Reads raw / writes processed", "file I/O")
    Rel(pipeline, artifacts, "Writes trained model", "file write")
    Rel(datasource, data, "Curated imports", "manual")
```

### Level 3 — Component

```mermaid
flowchart LR
    subgraph Web["Streamlit UI Container"]
        Inputs["InputForm"]
        Result["ResultPanel"]
        Charts["ChartPanel"]
        Report["ReportExporter"]
    end

    subgraph Pipeline["Training Pipeline Container"]
        Ingest["Ingestor"]
        Clean["Cleaner"]
        Feat["FeatureEngineer"]
        Train["Trainer"]
        Eval["Evaluator"]
        Persist["ArtifactStore"]
    end

    Inputs --> Result
    Result --> Charts
    Result --> Report

    Ingest --> Clean --> Feat --> Train --> Eval --> Persist
    Train <-->|save/load| Persist

    classDef web fill:#ef476f,color:#fff,stroke:#d62828;
    classDef pipe fill:#06d6a0,color:#000,stroke:#1b9e8a,stroke-width:2px;
    class Inputs,Result,Charts,Report web;
    class Ingest,Clean,Feat,Train,Eval,Persist pipe;
```

---

## 🧬 Class Diagram

```mermaid
classDiagram
    class HouseData {
        +DataFrame df
        +load(path: str) DataFrame
        +info()
        +describe()
    }

    class Preprocessor {
        -scaler: StandardScaler
        -encoder: OneHotEncoder
        +clean(df: DataFrame) DataFrame
        +engineer_features(df) DataFrame
        +split(X, y, test_size, seed) tuple
        +transform(X) ndarray
    }

    class ModelTrainer {
        -model: RegressorMixin
        +train(X_train, y_train) RegressorMixin
        +set_algorithm(name: str) void
        +cross_validate(X, y, k: int) dict
    }

    class Evaluator {
        +mae(y_true, y_pred) float
        +mse(y_true, y_pred) float
        +rmse(y_true, y_pred) float
        +r2(y_true, y_pred) float
        +report(y_true, y_pred) dict
    }

    class ArtifactStore {
        +save(model, path) void
        +load(path) RegressorMixin
        +version(name) str
    }

    class Predictor {
        -model: RegressorMixin
        -preprocessor: Preprocessor
        +predict(features: dict) float
        +confidence(features: dict) tuple
    }

    class StreamlitApp {
        +render_inputs() dict
        +render_result(price: float) void
        +render_charts() void
        +export_report() bytes
    }

    HouseData --> Preprocessor : provides df
    Preprocessor --> ModelTrainer : provides X/y
    ModelTrainer --> Evaluator : provides y_pred
    ModelTrainer --> ArtifactStore : persists model
    ArtifactStore --> Predictor : supplies model
    Predictor --> StreamlitApp : serves inference
    Preprocessor --> Predictor : provides transform
```

---

## 🗺️ Wardley Map

Maps each component on **Evolution (Genesis → Custom → Product → Commodity)** × **Value Chain (visible to user → hidden infrastructure)**.

```mermaid
flowchart LR
    subgraph Map["Wardley Map — House Price Predictor"]
        direction LR

        User(["👤 User Need:<br/>Instant price estimate"])

        UIViz["Streamlit UI"]
        Pred["Predictor service"]
        Charts["Charts / Reports"]
        Location["Location feature"]
        Forest["Random Forest"]
        Linear["Linear Regression"]
        Data["Curated dataset"]
        scikit["scikit-learn"]
        Stream["Streamlit framework"]
        PyRuntime["Python runtime"]
        Cloud["Cloud hosting"]

        User --> UIViz
        User --> Charts
        UIViz --> Pred
        UIViz --> Location
        Pred --> Linear
        Pred --> Forest
        Pred --> Data
        Linear --> scikit
        Forest --> scikit
        UIViz --> Stream
        Stream --> PyRuntime
        scikit --> PyRuntime
        Cloud --> Stream
    end

    classDef genesis fill:#ffe0b3,color:#000,stroke:#e59400;
    classDef custom fill:#ffd166,color:#000,stroke:#d4a017;
    classDef product fill:#a8dadc,color:#000,stroke:#457b9d;
    classDef commodity fill:#bde0bd,color:#000,stroke:#4a8b3a;

    class Location,Forest genesis;
    class Charts,Pred,Data custom;
    class scikit,UIViz product;
    class Linear,Stream,PyRuntime,Cloud commodity;
```

**Interpretation**

- *Genesis/Custom* (Location, Random Forest, Predictor): high differentiation — invest here for competitive edge.
- *Product* (scikit-learn, Streamlit UI): standardized building blocks — adopt, don't reinvent.
- *Commodity* (Python, cloud hosting, linear regression): use as utilities; minimize effort.

---

## 🔄 Data Flow

```mermaid
flowchart LR
    A[("Raw CSV")] -->|pandas.read_csv| B[DataFrame]
    B --> C{Missing values?}
    C -->|yes| D[Impute / drop]
    C -->|no| E[Encode categoricals]
    D --> E
    E --> F[Scale numerics]
    F --> G[train_test_split]
    G --> H[Train model]
    H --> I[Evaluate]
    I --> J{R² ≥ 0.80?}
    J -->|no| K[Tune / add features]
    K --> H
    J -->|yes| L[Persist artifact]
    L --> M[Serve via Streamlit]

    style A fill:#118ab2,color:#fff
    style M fill:#06d6a0,color:#000
    style K fill:#ffd166,color:#000
```

---

## 🧩 Components & Responsibilities

| Component        | File             | Responsibility |
|------------------|------------------|----------------|
| HouseData        | `preprocess.py`  | Load & sanity-check dataset |
| Preprocessor     | `preprocess.py`  | Clean, encode, scale, split |
| ModelTrainer     | `train.py`       | Fit regressor, cross-validate |
| Evaluator        | `evaluate.py`    | Compute MAE/MSE/RMSE/R² |
| ArtifactStore    | `model.py`       | Persist & version models |
| Predictor        | `model.py`       | Wrap model + preprocess for inference |
| StreamlitApp     | `app.py`         | Render inputs, results, charts, reports |

---

## #️⃣ Design Decisions (ADR-style)

### ADR-001 — Linear Regression as baseline
**Status:** Accepted · **Context:** Beginner-friendly model with interpretable coefficients. **Decision:** Ship Linear Regression first; add Random Forest behind a toggle. **Consequences:** Easier debugging, fast training, transparent feature weights.

### ADR-002 — Streamlit over Flask/FastAPI
**Status:** Accepted · **Context:** Single-author project needing instant UI. **Decision:** Streamlit for both demo and production preview. **Consequences:** Rapid prototyping; limited control over routing — acceptable at current scale.

### ADR-003 — Filesystem artifact store
**Status:** Accepted · **Context:** Small team, low write frequency. **Decision:** Use `.joblib` files under `models/`. **Consequences:** No DB ops overhead; promote to MLflow/registry if scale grows.

### ADR-004 — Optional ZenML pipeline
**Status:** Proposed · **Context:** Reproducibility & audit needs grow. **Decision:** Wrap training steps in ZenML pipelines; keep `train.py` as a thin entrypoint. **Consequences:** Slight onboarding cost, much better traceability.

---

## 📈 Non-Functional Requirements

| Attribute     | Target |
|---------------|--------|
| Latency       | Single prediction < 200 ms (P95) |
| Throughput    | ≥ 10 concurrent users on 1 vCPU |
| Availability  | 99% (Streamlit Community Cloud) |
| Reproducibility | Deterministic with fixed seeds |
| Maintainability | ≥ 80% unit-test coverage on logic layer |
| Security      | No PII stored; input validation on all fields |

---

## ⚠️ Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | Dataset too small → overfit | High | High | Cross-validation, regularization, synthetic data |
| 2 | Location categories unseen at inference | Med | Med | `handle_unknown='ignore'` in encoder + fallback mean |
| 3 | Concept drift in market prices | Med | High | Periodic retraining schedule + monitoring |
| 4 | Streamlit scale ceiling | Low | Med | Migrate inference API to FastAPI if needed |
| 5 | Dependency breakage | Low | Med | Pin versions; run CI matrix on Python 3.10/3.11 |
```
