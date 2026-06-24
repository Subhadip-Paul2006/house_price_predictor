# 🔧 TRD.md — Technical Requirements Document

> **Project:** House Price Prediction App
> **Owner:** ML Engineering
> **Status:** Draft v1.0
> **Last updated:** 2026-06-23

---

## 📑 Table of Contents

1. [Purpose & Scope](#-purpose--scope)
2. [Technical Objectives](#-technical-objectives)
3. [Functional Requirements](#-functional-requirements)
4. [Non-Functional Requirements](#-non-functional-requirements)
5. [System Sequence Diagrams](#-system-sequence-diagrams)
6. [Detailed Sequence — Prediction](#-detailed-sequence--prediction)
7. [Detailed Sequence — Training](#-detailed-sequence--training)
8. [Git Workflow Diagram](#-git-workflow-diagram)
9. [ZenML Pipeline](#-zenml-pipeline)
10. [Technology Radar](#-technology-radar)
11. [Model Performance XY Charts](#-model-performance-xy-charts)
12. [Error Budget & SLOs](#-error-budget--slos)
13. [Observability](#-observability)
14. [Security & Compliance](#-security--compliance)
15. [Test Strategy](#-test-strategy)
16. [Deployment](#-deployment)
17. [Open Questions](#-open-questions)

---

## 🎯 Purpose & Scope

This TRD defines **how** the House Price Prediction App is engineered: the components, contracts, pipelines, version-control strategy, and quality gates that turn the PRD's product goals into a runnable, observable, reproducible system.

**In scope:** training pipeline, inference service (Streamlit), model artifact storage, evaluation, CI/CD.

**Out of scope:** payment integration, user accounts, multi-tenant isolation (deferred to v2).

---

## 🚀 Technical Objectives

| ID | Objective | Success Criterion |
|----|-----------|-------------------|
| TO-1 | Reproducible training | Same seed → byte-identical coefficients |
| TO-2 | Sub-second inference | P95 latency < 200 ms per prediction |
| TO-3 | Observable pipelines | Every run emits metrics + artifact lineage |
| TO-4 | Safe deployments | CI gates block PR if R² drops > 2 pts vs main |
| TO-5 | Portable artifacts | `.joblib` loads on any Python 3.10+ host |

---

## ✅ Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Load CSV via Pandas into a typed DataFrame | Must |
| FR-2 | Detect & handle missing values (impute/drop) | Must |
| FR-3 | Encode categorical `Location` (OneHot, `handle_unknown='ignore'`) | Must |
| FR-4 | Scale numeric features (StandardScaler) | Must |
| FR-5 | Split train/test 80/20 with fixed seed | Must |
| FR-6 | Train Linear Regression baseline | Must |
| FR-7 | Optional Random Forest behind toggle | Should |
| FR-8 | Compute MAE, MSE, RMSE, R² | Must |
| FR-9 | Persist model + metadata to `models/` | Must |
| FR-10 | Expose Streamlit form + predict button | Must |
| FR-11 | Render prediction confidence band | Should |
| FR-12 | Download PDF estimate report | Should |
| FR-13 | Compare two models side-by-side | Could |

---

## 📐 Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| Performance | Inference P95 < 200 ms; cold start < 3 s |
| Scalability | 10 concurrent users on 1 vCPU Streamlit tier |
| Reliability | 99% monthly availability |
| Maintainability | ≥ 80% line coverage on logic modules |
| Reproducibility | All randomness seeded; pipeline deterministic |
| Portability | Runs on Win/macOS/Linux Python 3.10+ |
| Security | Input bounds validated; no remote code paths |
| Observability | Structured logs + run metrics per training |

---

## 🔁 System Sequence Diagrams

### High-level end-to-end flow

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant UI as Streamlit UI
    participant API as Predictor
    participant DB as Model Store
    participant SRC as Dataset

    U->>UI: Opens app, fills form
    UI->>API: predict(features)
    API->>DB: load latest model
    DB-->>API: model.pkl
    API->>API: transform + predict
    API-->>UI: price + confidence
    UI-->>U: render result + charts

    Note over UI,SRC: (Offline) Training path
    UI->>SRC: trigger retrain
    SRC->>API: ingest + train
    API->>DB: save new artifact
```

---

## 🔍 Detailed Sequence — Prediction

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant UI as StreamlitApp
    participant PP as Preprocessor
    participant P as Predictor
    participant AS as ArtifactStore

    U->>UI: Enter Area=1800, Beds=3, Baths=2, Age=5
    UI->>UI: validate inputs (bounds, types)
    UI->>P: predict({area, beds, baths, age, location})
    P->>AS: load("models/linear_regression.joblib")
    AS-->>P: (model, preprocessor)
    P->>PP: transform(features)
    PP->>PP: impute → encode → scale
    PP-->>P: X_vec (ndarray)
    P->>P: model.predict(X_vec)
    P-->>UI: (price=75.0 L, lo=71.0, hi=79.0)
    UI->>UI: render cards + chart
    UI-->>U: Estimated ₹75 L (±4 L)
```

---

## 🔍 Detailed Sequence — Training

```mermaid
sequenceDiagram
    autonumber
    participant Dev as Developer
    participant TP as TrainPipeline
    participant HD as HouseData
    participant PP as Preprocessor
    participant MT as ModelTrainer
    participant EV as Evaluator
    participant AS as ArtifactStore

    Dev->>TP: python train.py --model linear
    TP->>HD: load("house_data.csv")
    HD-->>TP: DataFrame
    TP->>PP: clean + engineer + split
    PP-->>TP: (X_train, X_test, y_train, y_test)
    TP->>MT: train(X_train, y_train, seed=42)
    MT-->>TP: fitted model
    TP->>EV: evaluate(model, X_test, y_test)
    EV-->>TP: {mae, mse, rmse, r2}
    TP->>AS: save(model, metrics, run_id)
    AS-->>TP: artifact path
    TP-->>Dev: "Training complete · R²=0.84"
```

---

## 🌿 Git Workflow Diagram

```mermaid
gitGraph
    commit id: "init"
    commit id: "scaffold"
    branch develop
    checkout develop
    commit id: "eda notebook"
    branch feat/preprocessing
    commit id: "clean+encode"
    commit id: "scale+split"
    checkout develop
    merge feat/preprocessing
    branch feat/linear-model
    commit id: "train LR"
    commit id: "eval metrics"
    checkout develop
    merge feat/linear-model
    branch feat/streamlit-ui
    commit id: "form + predict"
    commit id: "charts + report"
    checkout develop
    merge feat/streamlit-ui
    checkout main
    merge develop tag: "v0.1.0"
    branch feat/random-forest
    commit id: "rf baseline"
    commit id: "model toggle"
    checkout develop
    merge feat/random-forest
    checkout main
    merge develop tag: "v0.2.0"
```

**Branching rules**

- `main` — always deployable, tagged releases only.
- `develop` — integration branch.
- `feat/*`, `fix/*`, `docs/*` — short-lived, PR into `develop`.
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).

---

## 🧪 ZenML Pipeline

ZenML orchestrates the training steps for reproducibility and lineage.

```mermaid
flowchart LR
    subgraph Zen["ZenML Stack"]
        A[("raw CSV")] --> B[ingest_data_step]
        B --> C[clean_data_step]
        C --> D[engineer_features_step]
        D --> E[split_data_step]
        E --> F[train_model_step]
        F --> G[evaluate_model_step]
        G --> H[persist_artifact_step]
    end

    H --> I[("artifact store")]
    G -.metrics.-> J[(metadata store)]
    G -.lineage.-> K[(dashboard)]

    classDef step fill:#06d6a0,color:#000,stroke:#1b9e8a,stroke-width:2px;
    classDef store fill:#118ab2,color:#fff,stroke:#073b4c;
    class B,C,D,E,F,G,H step;
    class A,I,J,K store;
```

**Example skeleton**

```python
from zenml import pipeline, step

@step
def ingest_data(path: str):
    import pandas as pd
    return pd.read_csv(path)

@step
def train_model(X_train, y_train):
    from sklearn.linear_model import LinearRegression
    return LinearRegression().fit(X_train, y_train)

@pipeline
def house_price_pipeline(path: str):
    df = ingest_data(path)
    # ... clean, split ...
    model = train_model(X_train, y_train)
    # evaluate + persist

if __name__ == "__main__":
    house_price_pipeline.with_options(...).run("house_data.csv")
```

Run with `zenml up` to view lineage in the dashboard.

---

## 📡 Technology Radar

```mermaid
radar-beta
    title Tech Radar
    axis data["Tools"], model["Frameworks"], ops["MLOps"], ui["Front-end"]
    curve{"Adopt", "Trial", "Assess", "Hold"}

    Adopt("scikit-learn"), Adopt("Pandas"), Adopt("Streamlit"), Adopt("joblib")
    Trial("ZenML"), Trial("Random Forest")
    Assess("MLflow"), Assess("FastAPI inference"), Assess("Docker")
    Hold("Flask UI"), Hold("Pickle (prefer joblib)")
```

**Ring meanings**

- **Adopt** — proven, default choice.
- **Trial** — promising, used in non-critical path.
- **Assess** — exploring in spikes.
- **Hold** — avoid for new work.

---

## 📊 Model Performance XY Charts

```mermaid
xychart-beta
    title "R² vs. Training Data Size"
    x-axis ["100", "500", "1k", "5k", "10k", "50k"]
    y-axis "R² Score" 0 --> 1
    line [0.55, 0.71, 0.78, 0.84, 0.87, 0.90]
```

```mermaid
xychart-beta
    title "MAE (Lakh ₹) vs. Feature Count"
    x-axis ["3", "4", "5", "6", "7", "8"]
    y-axis "MAE" 0 --> 12
    line [9.5, 7.8, 6.2, 5.4, 5.0, 4.8]
```

```mermaid
xychart-beta
    title "Inference Latency vs. Concurrency"
    x-axis ["1", "5", "10", "25", "50", "100"]
    y-axis "Latency ms" 0 --> 1000
    line [45, 60, 110, 320, 720, 950]
```

---

## 🎯 Error Budget & SLOs

| SLO | Target | Error Budget (30d) |
|-----|--------|--------------------|
| Availability | 99.0% | 432 min downtime |
| Inference P95 latency | < 200 ms | 5% of requests may exceed |
| Training run success | ≥ 95% | 1 failure per 20 runs allowed |

When the budget is exhausted, feature work freezes until reliability is restored.

---

## 👁️ Observability

- **Logs:** Python `logging` → JSON to stdout; Streamlit Cloud captures automatically.
- **Metrics:** Every training run writes `metrics.json` (mae, mse, rmse, r2, n_rows) alongside artifact.
- **Lineage:** ZenML dashboard (or MLflow if adopted) links dataset → model → metrics.
- **Health:** `/healthz`-equivalent: a hidden Streamlit page pinging the model artifact.

---

## 🔐 Security & Compliance

- No PII collected; the dataset is public/synthetic.
- Input validation: bounds-checked (area > 0, beds ∈ [0, 10], etc.).
- Model files served read-only; writes restricted to CI.
- Dependencies scanned via `pip-audit` in CI.

---

## 🧪 Test Strategy

| Layer | Tool | Coverage target |
|-------|------|-----------------|
| Unit (preprocess, metrics) | pytest | 80% |
| Model contract | pytest + sklearn `check_estimator` | key regressors |
| Integration (train→save→load→predict) | pytest | happy path + 1 failure |
| UI smoke | Playwright (optional) | form submit |
| Regression gate | CI script | R² must not drop > 2 pts |

```python
# sample unit
def test_mae_zero_when_perfect():
    from evaluate import mae
    assert mae([10, 20], [10, 20]) == 0.0
```

---

## 🚢 Deployment

```mermaid
flowchart LR
    Dev([Developer]) -->|push| GH[(GitHub)]
    GH -->|on PR| CI[GitHub Actions]
    CI -->|lint + test| RES{Pass?}
    RES -->|no| Dev
    RES -->|yes| Merge[merge to develop]
    Merge -->|tag main| Rel[Release vX.Y.Z]
    Rel -->|build| IMG[Docker image]
    IMG -->|deploy| Cloud[Streamlit Cloud / Fly.io]
    Rel -->|attach| Artifacts[(model artifacts)]
```

---

## ❓ Open Questions

1. Do we need user accounts for the report-download feature (v2)?
2. Should the Location feature use geocoding or a fixed city list?
3. MLflow vs. ZenML-only for experiment tracking — decision needed by sprint 3.
4. Acceptable R² threshold to auto-promote a model to production?
```
