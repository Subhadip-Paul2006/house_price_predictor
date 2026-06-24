# 📋 PRD.md — Product Requirements Document

> **Product:** House Price Prediction App
> **Author:** Product Team
> **Status:** Draft v1.0
> **Last updated:** 2026-06-23

---

## 📑 Table of Contents

1. [Product Vision](#-product-vision)
2. [Target Users & Personas](#-target-users--personas)
3. [Problem Statement](#-problem-statement)
4. [Goals & Non-Goals](#-goals--non-goals)
5. [User Journey Diagram](#-user-journey-diagram)
6. [User Story Map / Mindmap](#-user-story-map--mindmap)
7. [Gantt Chart](#-gantt-chart)
8. [Project Timeline](#-project-timeline)
9. [Kanban Board](#-kanban-board)
10. [Feature Prioritization Quadrants](#-feature-prioritization-quadrants)
11. [Success Metrics](#-success-metrics)
12. [User Stories & Acceptance Criteria](#-user-stories--acceptance-criteria)
13. [Out of Scope](#-out-of-scope)
14. [Risks & Assumptions](#-risks--assumptions)
15. [Launch Plan](#-launch-plan)

---

## 🌅 Product Vision

> *Let anyone estimate a fair house price in seconds — no spreadsheets, no broker calls, no guesswork.*

The House Price Prediction App turns a handful of property details into a trustworthy price estimate, powered by a transparent machine-learning model and wrapped in a clean, friendly web UI.

---

## 👥 Target Users & Personas

```mermaid
mindmap
  root((Personas))
    First-time Buyer
      wants affordability check
      low ML literacy
      values clarity
    Real-estate Enthusiast
      compares neighborhoods
      wants charts
      power user
    ML Learner
      studies the pipeline
      wants code access
      forking the repo
    Seller
      sanity-check listing price
      wants fast estimate
      mobile-first
```

| Persona | Goal | Pain we solve |
|---------|------|---------------|
| 🧑‍🎓 **First-time Buyer (Priya)** | Affordability check before viewing | Instant estimate + confidence band |
| 🏘️ **Enthusiast (Arjun)** | Compare 2-3 neighborhoods | Side-by-side comparison + charts |
| 🎓 **ML Learner (Sam)** | Study a clean end-to-end project | Open-source, documented pipeline |
| 🏠 **Seller (Meera)** | Validate her listing price | One-screen estimate in seconds |

---

## ❗ Problem Statement

Real-estate prices feel opaque. Buyers overpay, sellers underprice, and learners struggle to find a *complete* ML project they can run end-to-end. Existing tools are either black-box portals or fragmented tutorials.

**Why now?** Public datasets, mature libraries (scikit-learn, Streamlit), and free hosting make a transparent, runnable estimator achievable in days, not months.

---

## 🎯 Goals & Non-Goals

**Goals**
- Deliver an accurate (R² ≥ 0.80) price estimate from ≤ 6 inputs in < 3 seconds.
- Provide a one-command local setup and a public demo URL.
- Ship a documented, forkable codebase suitable as a portfolio project.

**Non-Goals**
- Not a brokerage or transaction platform.
- No user accounts or saved estimates in v1.
- No real-time market data feeds (v1 uses static dataset).

---

## 🛤️ User Journey Diagram

```mermaid
journey
    title First-time Buyer's Journey
    section Discover
      Search "house price estimator": 5: Priya
      Land on app URL: 5: Priya
    section Input
      Read field hints: 4: Priya
      Enter area & beds: 4: Priya
      Pick location: 3: Priya
    section Predict
      Click Predict: 5: Priya
      See estimated price: 5: Priya
      View confidence band: 4: Priya
    section Decide
      Compare second property: 4: Priya
      Download PDF report: 3: Priya
      Share with family: 4: Priya
```

```mermaid
journey
    title ML Learner's Journey
    section Explore
      Read README: 5: Sam
      Clone repo: 5: Sam
    section Run
      Install deps: 4: Sam
      python train.py: 4: Sam
      streamlit run app.py: 5: Sam
    section Learn
      Open notebooks/: 5: Sam
      Tweak features: 4: Sam
      Fork & extend: 5: Sam
```

---

## 🗺️ User Story Map / Mindmap

```mermaid
mindmap
  root((House Price App))
    Discover
      SEO landing
      Demo link
      GitHub README
    Input
      Area slider
      Bedrooms
      Bathrooms
      Age
      Location dropdown
    Predict
      Instant estimate
      Confidence band
      Comparable chart
    Compare
      Two properties
      Neighborhood view
    Export
      PDF report
      Shareable link
    Learn
      Notebooks
      Source code
      Docs
```

---

## 📅 Gantt Chart

```mermaid
gantt
    title House Price Predictor — Delivery Plan (8 weeks)
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Discovery
    Research & dataset selection       :a1, 2026-06-23, 7d
    Personas & PRD draft               :a2, after a1, 4d

    section Data
    Acquire & clean dataset            :b1, after a2, 5d
    EDA notebook                       :b2, after b1, 4d

    section Modeling
    Baseline Linear Regression         :c1, after b2, 4d
    Evaluation harness                 :c2, after c1, 3d
    Random Forest + toggle             :c3, after c2, 5d

    section App
    Streamlit UI scaffold              :d1, after c2, 4d
    Confidence band + charts           :d2, after d1, 4d
    PDF report                         :d3, after d2, 3d

    section Hardening
    Unit + integration tests           :e1, after d3, 4d
    CI pipeline                        :e2, after e1, 3d
    Docs & screenshots                 :e3, after e2, 3d

    section Launch
    Deploy to cloud                    :f1, after e3, 3d
    Beta feedback                      :f2, after f1, 5d
    v1.0 release                       :milestone, f3, after f2, 0d
```

---

## 🗓️ Project Timeline

```mermaid
timeline
    title Project Timeline
    section Week 1 — Discovery
        Dataset shortlisted : 2026-06-23
        PRD signed off      : 2026-06-27
    section Week 2 — Data
        EDA complete        : 2026-07-04
    section Week 3-4 — Modeling
        Linear baseline     : 2026-07-11
        R² ≥ 0.80 hit       : 2026-07-14
        Random Forest       : 2026-07-18
    section Week 5-6 — App
        Streamlit MVP       : 2026-07-22
        Charts + report     : 2026-07-29
    section Week 7 — Hardening
        Tests green         : 2026-08-04
        CI live             : 2026-08-07
    section Week 8 — Launch
        Cloud deploy        : 2026-08-12
        v1.0 release        : 2026-08-18
```

---

## 🧱 Kanban Board

```mermaid
flowchart LR
    BACKLOG["📥 Backlog"]:::col
    TODO["🗓️ To Do"]:::col
    DOING["🔨 In Progress"]:::col
    REVIEW["👀 Review"]:::col
    DONE["✅ Done"]:::col

    BACKLOG --> TODO --> DOING --> REVIEW --> DONE

    BACKLOG -.- B1["Map integration"]
    BACKLOG -.- B2["Analytics dashboard"]
    BACKLOG -.- B3["Multi-model ensemble"]

    TODO -.- T1["Confidence band UI"]
    TODO -.- T2["PDF report"]

    DOING -.- D1["Random Forest training"]

    REVIEW -.- R1["Location encoding PR"]
    REVIEW -.- R2["Streamlit charts PR"]

    DONE -.- Z1["Linear baseline"]
    DONE -.- Z2["EDA notebook"]
    DONE -.- Z3["Preprocessing"]
    DONE -.- Z4["Evaluation metrics"]

    classDef col fill:#1d3557,color:#fff,stroke:#0d1b2a,stroke-width:2px;
```

**Current sprint snapshot**

| Backlog | To Do | In Progress | Review | Done |
|---------|-------|-------------|--------|------|
| Map integration | Confidence band UI | Random Forest training | Location encoding PR | Linear baseline |
| Analytics dashboard | PDF report | | Streamlit charts PR | EDA notebook |
| Multi-model ensemble | | | | Preprocessing |
| | | | | Evaluation metrics |

---

## 🎯 Feature Prioritization Quadrants

### Impact vs. Effort

```mermaid
quadrantChart
    title Impact vs. Effort
    x-axis Low Impact --> High Impact
    y-axis Low Effort --> High Effort
    quadrant-1 Do Next
    quadrant-2 Schedule
    quadrant-3 Backlog
    quadrant-4 Avoid
    "Core prediction": [0.92, 0.30]
    "Clean UI": [0.82, 0.25]
    "Charts": [0.66, 0.30]
    "Compare models": [0.74, 0.45]
    "Location pricing": [0.86, 0.55]
    "Download report": [0.55, 0.35]
    "Confidence band": [0.60, 0.50]
    "Map integration": [0.93, 0.86]
    "Dashboard": [0.88, 0.82]
```

### User Value vs. Differentiation

```mermaid
quadrantChart
    title User Value vs. Differentiation
    x-axis Low Differentiation --> High Differentiation
    y-axis Low User Value --> High User Value
    quadrant-1 Differentiate
    quadrant-2 Maintain
    quadrant-3 Decline
    quadrant-4 Re-think
    "Core prediction": [0.30, 0.85]
    "Clean UI": [0.35, 0.78]
    "Charts": [0.40, 0.65]
    "Location pricing": [0.70, 0.84]
    "Confidence band": [0.65, 0.62]
    "Compare models": [0.60, 0.58]
    "Map integration": [0.90, 0.88]
    "Analytics dashboard": [0.85, 0.80]
    "Report download": [0.50, 0.55]
```

---

## 📏 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Model R² | ≥ 0.80 | `train.py` eval output |
| Prediction latency | P95 < 200 ms | Streamlit logs |
| Demo monthly active users | ≥ 200 | Hosting analytics |
| GitHub stars (6 mo) | ≥ 50 | GitHub |
| Time-to-first-prediction | < 3 min from clone | Manual test |
| Test coverage | ≥ 80% | pytest/coverage CI |

---

## 📖 User Stories & Acceptance Criteria

**US-1 — Get a price estimate** *(Must)*
> As a buyer, I want to enter house details and see an estimated price.
- ✅ Inputs: area, beds, baths, age, location
- ✅ Result renders in < 1 s after click
- ✅ Price displayed in Lakh ₹ / Cr

**US-2 — See confidence** *(Should)*
> As a buyer, I want a confidence band so I trust the estimate.
- ✅ Shows ± range based on residual std
- ✅ Visible on chart

**US-3 — Compare properties** *(Should)*
> As an enthusiast, I want to compare two estimates side-by-side.
- ✅ Two input panels
- ✅ Comparison bar chart

**US-4 — Download report** *(Should)*
> As a seller, I want a PDF of the estimate.
- ✅ One-click PDF
- ✅ Includes inputs + estimate + date

**US-5 — Switch models** *(Could)*
> As a learner, I want to toggle Linear vs. Random Forest.
- ✅ Dropdown switches model
- ✅ Metric badge shown

**US-6 — Fork & extend** *(Could)*
> As an ML learner, I want clean docs to fork.
- ✅ README + notebooks + ARCHITECTURE.md

---

## 🚫 Out of Scope (v1)

- User accounts / saved history
- Real-time MLS / market data feeds
- In-app payments or referrals
- Mobile native apps (responsive web only)
- Multi-language UI

---

## ⚠️ Risks & Assumptions

| # | Risk / Assumption | Mitigation |
|---|-------------------|------------|
| 1 | Dataset coverage skews to one city → poor generalization | Disclose region; expand data in v2 |
| 2 | Users may treat estimate as appraisal | Prominent disclaimer |
| 3 | Model drift as markets change | Quarterly retraining schedule |
| 4 | Free-hosting scale limits | Plan migration to Fly.io if needed |
| 5 | Location categories unseen at inference | Fallback to city mean |

---

## 🚀 Launch Plan

```mermaid
flowchart LR
    Dev([Internal dev done]) --> Alpha[Closed alpha<br/>5 users]
    Alpha --> Feedback{Feedback positive?}
    Feedback -->|no| Fix[Iterate]
    Fix --> Alpha
    Feedback -->|yes| Beta[Public beta<br/>demo URL]
    Beta --> Polish[Docs + screenshots]
    Polish --> GA([🎉 v1.0 release])
    GA --> Post[Monitor metrics<br/>plan v1.1]
```

**Launch checklist**

- [ ] R² ≥ 0.80 on held-out test set
- [ ] Inference P95 < 200 ms
- [ ] README + screenshots + demo GIF
- [ ] Public demo URL live
- [ ] LICENSE + CONTRIBUTING added
- [ ] GitHub release v1.0 tagged
