<![CDATA[<div align="center">

# 📋 Product Requirements Document (PRD)

### House Price Predictor — v1.0

</div>

---

## 1. Product Overview

| Field | Detail |
|-------|--------|
| **Product Name** | House Price Predictor |
| **Version** | 1.0.0 |
| **Author** | Subhadip Paul |
| **Date** | June 2026 |
| **Status** | ✅ Complete |

### Problem Statement

Estimating property prices in India requires consulting multiple brokers. This tool provides **instant, ML-powered price estimates** with confidence intervals.

---

## 2. User Stories

```mermaid
flowchart LR
    subgraph US-1 ["US-1: Single Prediction"]
        A1["Enter property details → get price"]
    end
    subgraph US-2 ["US-2: Confidence Band"]
        A2["See 95% confidence interval"]
    end
    subgraph US-3 ["US-3: Comparison"]
        A3["Compare two properties"]
    end
    subgraph US-4 ["US-4: PDF Report"]
        A4["Download estimate as PDF"]
    end
    subgraph US-5 ["US-5: Model Switch"]
        A5["Toggle LR / RF model"]
    end
    US-1 --> US-2 --> US-3 --> US-4 --> US-5
    style US-1 fill:#4CAF50,color:#fff
    style US-2 fill:#2196F3,color:#fff
    style US-3 fill:#FF9800,color:#fff
    style US-4 fill:#9C27B0,color:#fff
    style US-5 fill:#F44336,color:#fff
```

---

## 3. Feature Requirements

### Input Features

| # | Feature | Type | Range |
|---|---------|------|-------|
| 1 | Area (sq ft) | Numeric | 300 – 6,000 |
| 2 | Bedrooms | Integer | 1 – 10 |
| 3 | Bathrooms | Integer | 1 – 8 |
| 4 | Age (years) | Integer | 0 – 50 |
| 5 | Location | Categorical | Downtown / Urban / Suburban / Rural |

### Output Features

| # | Output | Description |
|---|--------|-------------|
| 1 | Estimated Price | Point estimate in Lakhs ₹ |
| 2 | Confidence Band | 95% CI lower & upper bounds |
| 3 | Model Metrics | R², MAE, RMSE, CV R² |
| 4 | PDF Report | Downloadable summary |

---

## 4. User Flow

```mermaid
flowchart TD
    Start([User Opens App]) --> Sidebar[Configure Sidebar]
    Sidebar --> Model{Select Model}
    Model --> Input[Enter Property Details]
    Input --> Tab{Choose Tab}
    Tab -->|Predict| Predict[Click Predict → View Estimate]
    Tab -->|Compare| Compare[Enter Two Props → Compare]
    Tab -->|Charts| Charts[View Scatter + Importance]
    Predict --> PDF[Download PDF Report]
    style Start fill:#4CAF50,color:#fff
    style Predict fill:#2196F3,color:#fff
    style Compare fill:#FF9800,color:#fff
    style Charts fill:#9C27B0,color:#fff
```

---

## 5. Acceptance Criteria

| User Story | Criteria |
|------------|----------|
| US-1 | Displays numeric price ≥ 0 in Lakhs ₹ |
| US-2 | Shows lower/upper bound using 1.96 × residual_std |
| US-3 | Side-by-side columns with grouped bar chart |
| US-4 | PDF downloads with property details + metrics |
| US-5 | Radio button toggles model, prediction updates |

---

## 6. Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Prediction Latency | < 500ms |
| Model R² | ≥ 0.89 |
| CV R² | ≥ 0.85 |
| Python | 3.10+ |

---

<div align="center">

_PRD v1.0 — House Price Predictor_

</div>
]]>
