
# 📖 USAGE GUIDE

### House Price Predictor — Complete Setup & Contribution Guide

[![Fork](https://img.shields.io/badge/Step_1-Fork_Repo-4CAF50?style=for-the-badge&logo=github)](https://github.com/Subhadip-Paul2006/house_price_predictor/fork)
[![Install](https://img.shields.io/badge/Step_2-Install_Packages-2196F3?style=for-the-badge&logo=python)](https://pip.pypa.io)
[![Run](https://img.shields.io/badge/Step_3-Run_Project-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Contribute](https://img.shields.io/badge/Step_4-Contribute-FF9800?style=for-the-badge&logo=git)](https://github.com/Subhadip-Paul2006/house_price_predictor/pulls)

</div>

---

## 📑 Table of Contents

- [Overview Workflow](#-overview-workflow)
- [Step 1 — Fork the Repository](#-step-1--fork-the-repository)
- [Step 2 — Install Virtual Environment & Packages](#-step-2--install-virtual-environment--packages)
- [Step 3 — Run the Project](#-step-3--run-the-project)
- [Step 4 — How Others Can Contribute](#-step-4--how-others-can-contribute)
- [Troubleshooting](#-troubleshooting)
- [Project Architecture Quick Reference](#-project-architecture-quick-reference)

---

## 🔄 Overview Workflow

The complete journey from forking to contributing:

```mermaid
flowchart LR
    A["🍴 Fork Repo"] --> B["📥 Clone Locally"]
    B --> C["🐍 Create venv"]
    C --> D["📦 Install Packages"]
    D --> E["📊 Generate Data"]
    E --> F["🤖 Train Models"]
    F --> G["🚀 Run App"]
    G --> H["✏️ Make Changes"]
    H --> I["🔀 Submit PR"]

    style A fill:#4CAF50,color:#fff
    style B fill:#4CAF50,color:#fff
    style C fill:#2196F3,color:#fff
    style D fill:#2196F3,color:#fff
    style E fill:#FF9800,color:#fff
    style F fill:#FF9800,color:#fff
    style G fill:#FF4B4B,color:#fff
    style H fill:#9C27B0,color:#fff
    style I fill:#9C27B0,color:#fff
```

---

## 🍴 Step 1 — Fork the Repository

### What is Forking?

Forking creates your own copy of the repository on GitHub, so you can experiment freely without affecting the original project.

### Fork Workflow

```mermaid
flowchart TD
    Original["🏠 Original Repo<br/>Subhadip-Paul2006/house_price_predictor"] 
    Original -->|"1. Click Fork ↗️"| Fork["🍴 Your Fork<br/>YOUR_USERNAME/house_price_predictor"]
    Fork -->|"2. Clone to PC"| Local["💻 Local Copy<br/>D:\house_price_predictor"]
    Local -->|"3. Add upstream"| Sync["🔗 Sync with Original"]

    style Original fill:#24292e,color:#fff
    style Fork fill:#4CAF50,color:#fff
    style Local fill:#2196F3,color:#fff
    style Sync fill:#FF9800,color:#fff
```

### 1.1 — Fork on GitHub

1. Go to the repository: **[github.com/Subhadip-Paul2006/house_price_predictor](https://github.com/Subhadip-Paul2006/house_price_predictor)**
2. Click the **Fork** button (top-right corner)
3. Select your GitHub account as the destination
4. Wait for GitHub to create your fork

### 1.2 — Clone Your Fork Locally

```bash
# Replace YOUR_USERNAME with your GitHub username
git clone https://github.com/YOUR_USERNAME/house_price_predictor.git

# Navigate into the project directory
cd house_price_predictor
```

### 1.3 — Add Upstream Remote (for staying in sync)

```bash
# Link to the original repo so you can pull future updates
git remote add upstream https://github.com/Subhadip-Paul2006/house_price_predictor.git

# Verify remotes
git remote -v
```

**Expected output:**

```
origin    https://github.com/YOUR_USERNAME/house_price_predictor.git (fetch)
origin    https://github.com/YOUR_USERNAME/house_price_predictor.git (push)
upstream  https://github.com/Subhadip-Paul2006/house_price_predictor.git (fetch)
upstream  https://github.com/Subhadip-Paul2006/house_price_predictor.git (push)
```

### 1.4 — Sync Your Fork (anytime)

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

---

## 🐍 Step 2 — Install Virtual Environment & Packages

### Why a Virtual Environment?

Virtual environments isolate your project's Python packages from the system-wide installation, preventing version conflicts.

### Environment Setup Flow

```mermaid
flowchart TD
    A["🐍 System Python 3.10+"] --> B["📁 Create Virtual Environment"]
    B --> C{"💻 Your OS?"}
    C -->|Windows| D["venv\\Scripts\\activate"]
    C -->|macOS / Linux| E["source venv/bin/activate"]
    D --> F["📦 pip install -r requirements.txt"]
    E --> F
    F --> G["✅ All Packages Installed"]

    subgraph Packages ["📦 Installed Packages"]
        P1["pandas ≥ 2.0"]
        P2["numpy ≥ 1.24"]
        P3["scikit-learn ≥ 1.4"]
        P4["streamlit ≥ 1.30"]
        P5["matplotlib ≥ 3.7"]
        P6["joblib ≥ 1.3"]
        P7["fpdf ≥ 1.7.2"]
        P8["seaborn ≥ 0.13"]
    end

    G --> Packages

    style A fill:#3776AB,color:#fff
    style B fill:#4CAF50,color:#fff
    style F fill:#2196F3,color:#fff
    style G fill:#FF9800,color:#fff
```

### 2.1 — Check Python Version

```bash
python --version
# Required: Python 3.10 or higher
```

> **⚠️ Note:** If you have both `python` and `python3`, use `python3` on macOS/Linux.

### 2.2 — Create Virtual Environment

```bash
# Create a virtual environment named 'venv'
python -m venv venv
```

### 2.3 — Activate the Virtual Environment

| OS | Command |
|----|---------|
| **Windows (CMD)** | `venv\Scripts\activate` |
| **Windows (PowerShell)** | `.\venv\Scripts\Activate.ps1` |
| **macOS / Linux** | `source venv/bin/activate` |

After activation, your terminal prompt will show `(venv)`:

```
(venv) D:\house_price_predictor>
```

### 2.4 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Package Dependency Map

```mermaid
flowchart LR
    subgraph Core ["🔬 Core ML"]
        sklearn["scikit-learn ≥ 1.4"]
        numpy["numpy ≥ 1.24"]
        pandas["pandas ≥ 2.0"]
    end

    subgraph Viz ["📊 Visualization"]
        matplotlib["matplotlib ≥ 3.7"]
        seaborn["seaborn ≥ 0.13"]
    end

    subgraph Web ["🌐 Web & Export"]
        streamlit["streamlit ≥ 1.30"]
        fpdf["fpdf ≥ 1.7.2"]
    end

    subgraph Util ["🔧 Utilities"]
        joblib["joblib ≥ 1.3"]
    end

    pandas --> numpy
    sklearn --> numpy
    seaborn --> matplotlib
    streamlit --> pandas
    streamlit --> matplotlib

    style Core fill:#E3F2FD
    style Viz fill:#FFF3E0
    style Web fill:#FFEBEE
    style Util fill:#E8F5E9
```

### 2.5 — Verify Installation

```bash
python -c "import streamlit, sklearn, pandas, matplotlib, fpdf; print('All packages installed successfully!')"
```

---

## 🚀 Step 3 — Run the Project

### Execution Pipeline

```mermaid
flowchart TD
    subgraph Phase1 ["Phase 1: Data Generation"]
        A["python generate_data.py"] --> B[("data/house_data.csv<br/>1000 rows created")]
    end

    subgraph Phase2 ["Phase 2: Model Training"]
        B --> C["python train.py"]
        C --> D["Linear Regression<br/>R² = 0.9018"]
        C --> E["Random Forest<br/>R² = 0.9313"]
        D --> F[("models/linear_regression/<br/>model.joblib + metrics.json")]
        E --> G[("models/random_forest/<br/>model.joblib + metrics.json")]
    end

    subgraph Phase3 ["Phase 3: Launch App"]
        F --> H["streamlit run app.py"]
        G --> H
        H --> I["🌐 http://localhost:8501"]
    end

    style Phase1 fill:#E8F5E9
    style Phase2 fill:#E3F2FD
    style Phase3 fill:#FFEBEE
```

### 3.1 — Generate the Dataset

> Skip this if `data/house_data.csv` already exists.

```bash
python generate_data.py
```

**Expected output:**

```
✅ Generated 1000 rows → data/house_data.csv
   Missing values:
Area         20
Bedrooms     10
Bathrooms     0
Age           0
Location     15
Price         0
```

### 3.2 — Train the Models

```bash
# Train BOTH models (Linear Regression + Random Forest)
python train.py

# Or train a specific model:
python train.py --model linear_regression
python train.py --model random_forest
```

**Expected training output:**

```
📊 Random Forest — Results
   Training samples : 800
   Test samples     : 200
   MAE              : 15.11 Lakh
   RMSE             : 20.77 Lakh
   R² (test)        : 0.9313
   R² (CV mean)     : 0.8816 ± 0.0352
   Residual Std     : 20.81 Lakh
```

### Training Pipeline Stages

```mermaid
flowchart LR
    S1["1️⃣ Load<br/>CSV"] --> S2["2️⃣ Clean<br/>Impute NaN"]
    S2 --> S3["3️⃣ Engineer<br/>Features"]
    S3 --> S4["4️⃣ Split<br/>80/20"]
    S4 --> S5["5️⃣ Encode<br/>Scale"]
    S5 --> S6["6️⃣ Train<br/>Model"]
    S6 --> S7["7️⃣ Cross<br/>Validate"]
    S7 --> S8["8️⃣ Evaluate<br/>Test Set"]
    S8 --> S9["9️⃣ Save<br/>Artifacts"]

    style S1 fill:#4CAF50,color:#fff
    style S2 fill:#4CAF50,color:#fff
    style S3 fill:#4CAF50,color:#fff
    style S4 fill:#2196F3,color:#fff
    style S5 fill:#2196F3,color:#fff
    style S6 fill:#FF9800,color:#fff
    style S7 fill:#FF9800,color:#fff
    style S8 fill:#9C27B0,color:#fff
    style S9 fill:#9C27B0,color:#fff
```

### 3.3 — Launch the Streamlit App

```bash
streamlit run app.py
```

**The app will open in your browser at `http://localhost:8501`.**

### App Features Overview

```mermaid
flowchart TD
    App["🏠 House Price Predictor<br/>http://localhost:8501"]

    App --> Sidebar["⚙️ Sidebar<br/>Model: LR / RF<br/>Area · Beds · Baths · Age · Location"]

    App --> Tab1["🔮 Predict Tab"]
    App --> Tab2["⚖️ Compare Tab"]
    App --> Tab3["📊 Charts Tab"]

    Tab1 --> R1["Estimated Price + 95% CI"]
    Tab1 --> R2["Model Metrics: R², MAE, RMSE"]
    Tab1 --> R3["📄 Download PDF Report"]

    Tab2 --> R4["Property A vs Property B"]
    Tab2 --> R5["Grouped Bar Chart"]

    Tab3 --> R6["📈 Area vs Price Scatter"]
    Tab3 --> R7["📊 Feature Importance"]

    style App fill:#FF4B4B,color:#fff
    style Tab1 fill:#667eea,color:#fff
    style Tab2 fill:#FF9800,color:#fff
    style Tab3 fill:#4CAF50,color:#fff
```

### 3.4 — Model Performance Summary

| Metric | Linear Regression | Random Forest |
|--------|:-----------------:|:-------------:|
| **R² Score** | 0.9018 | **0.9313** ✅ |
| **MAE** | 19.38 L | **15.11 L** ✅ |
| **RMSE** | 24.82 L | **20.77 L** ✅ |
| **CV R²** | 0.8782 | **0.8816** ✅ |

```mermaid
xychart-beta
    title "R² Score Comparison"
    x-axis ["Linear Regression", "Random Forest"]
    y-axis "R² Score" 0.85 --> 0.95
    bar [0.9018, 0.9313]
```

```mermaid
xychart-beta
    title "Error Comparison (Lower = Better)"
    x-axis ["MAE (Lakhs)", "RMSE (Lakhs)"]
    y-axis "Value" 0 --> 30
    bar [19.38, 24.82]
    bar [15.11, 20.77]
```

---

## 🤝 Step 4 — How Others Can Contribute

### Contribution Workflow

```mermaid
flowchart TD
    A["🍴 Fork the Repo"] --> B["📥 Clone Your Fork"]
    B --> C["🌿 Create Feature Branch"]
    C --> D["✏️ Make Changes"]
    D --> E["🧪 Test Your Changes"]
    E --> F{"Tests Pass?"}
    F -->|No| D
    F -->|Yes| G["💾 Commit Changes"]
    G --> H["📤 Push to Your Fork"]
    H --> I["🔀 Create Pull Request"]
    I --> J["👀 Code Review"]
    J --> K{"Approved?"}
    K -->|Changes Requested| D
    K -->|Yes| L["🎉 Merged!"]

    style A fill:#4CAF50,color:#fff
    style C fill:#2196F3,color:#fff
    style E fill:#FF9800,color:#fff
    style I fill:#9C27B0,color:#fff
    style L fill:#4CAF50,color:#fff
```

### 4.1 — Create a Feature Branch

**Never work directly on `main`.** Always create a feature branch:

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Examples:
git checkout -b feature/add-gradient-boosting
git checkout -b fix/prediction-error
git checkout -b docs/update-readme
```

### Branch Naming Convention

```mermaid
flowchart LR
    subgraph Convention ["📌 Branch Naming"]
        F["feature/* → New features"]
        B["fix/* → Bug fixes"]
        D["docs/* → Documentation"]
        R["refactor/* → Code refactoring"]
        T["test/* → Adding tests"]
    end

    style Convention fill:#E3F2FD
```

| Type | Pattern | Example |
|------|---------|---------|
| New Feature | `feature/description` | `feature/add-xgboost` |
| Bug Fix | `fix/description` | `fix/negative-price` |
| Documentation | `docs/description` | `docs/add-api-guide` |
| Refactor | `refactor/description` | `refactor/preprocess-pipeline` |
| Tests | `test/description` | `test/add-unit-tests` |

### 4.2 — Make Your Changes

Some ideas for contributions:

| Category | Idea | Difficulty |
|----------|------|:----------:|
| 🤖 **New Model** | Add XGBoost or Gradient Boosting | ⭐⭐ |
| 📊 **Visualization** | Add correlation heatmap | ⭐ |
| 🧪 **Testing** | Add pytest unit tests | ⭐⭐ |
| 📓 **Notebook** | Add EDA Jupyter notebook | ⭐ |
| 🎨 **UI** | Add dark mode toggle | ⭐⭐ |
| 📈 **Feature** | Add price trend over age chart | ⭐ |
| 🌐 **Deploy** | Deploy on Streamlit Cloud | ⭐⭐ |
| 📄 **Docs** | Improve inline docstrings | ⭐ |

### 4.3 — Test Your Changes

```bash
# 1. Verify the data pipeline still works
python generate_data.py

# 2. Verify training completes successfully
python train.py

# 3. Verify the app starts without errors
streamlit run app.py
```

### 4.4 — Commit and Push

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add XGBoost model support"

# Push to your fork
git push origin feature/your-feature-name
```

### Commit Message Format

| Prefix | Usage |
|--------|-------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Code refactoring |
| `test:` | Adding/updating tests |
| `style:` | Formatting, no code change |

### 4.5 — Create a Pull Request

1. Go to your fork on GitHub
2. Click **"Compare & pull request"**
3. Fill in the PR template:
   - **Title:** Clear, descriptive title
   - **Description:** What you changed and why
   - **Testing:** How you verified your changes
4. Submit the PR

### Pull Request Lifecycle

```mermaid
flowchart LR
    Draft["📝 Draft PR"] --> Open["🟢 Open PR"]
    Open --> Review["👀 Review"]
    Review --> Changes["🔄 Changes<br/>Requested"]
    Changes --> Update["✏️ Push Fixes"]
    Update --> Review
    Review --> Approve["✅ Approved"]
    Approve --> Merge["🎉 Merged"]

    style Draft fill:#E0E0E0
    style Open fill:#4CAF50,color:#fff
    style Review fill:#2196F3,color:#fff
    style Approve fill:#FF9800,color:#fff
    style Merge fill:#9C27B0,color:#fff
```

---

## 🔧 Troubleshooting

### Common Issues & Fixes

```mermaid
flowchart TD
    Issue1["❌ ModuleNotFoundError"] --> Fix1["pip install -r requirements.txt<br/>Check venv is activated"]
    Issue2["❌ FileNotFoundError:<br/>model.joblib"] --> Fix2["Run: python train.py"]
    Issue3["❌ FileNotFoundError:<br/>house_data.csv"] --> Fix3["Run: python generate_data.py"]
    Issue4["❌ Streamlit port in use"] --> Fix4["streamlit run app.py --server.port 8502"]
    Issue5["❌ Permission denied<br/>(PowerShell)"] --> Fix5["Set-ExecutionPolicy -Scope CurrentUser RemoteSigned"]

    style Issue1 fill:#FFCDD2
    style Issue2 fill:#FFCDD2
    style Issue3 fill:#FFCDD2
    style Issue4 fill:#FFCDD2
    style Issue5 fill:#FFCDD2
    style Fix1 fill:#C8E6C9
    style Fix2 fill:#C8E6C9
    style Fix3 fill:#C8E6C9
    style Fix4 fill:#C8E6C9
    style Fix5 fill:#C8E6C9
```

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Ensure venv is activated and run `pip install -r requirements.txt` |
| `FileNotFoundError: model.joblib` | Run `python train.py` to train models first |
| `FileNotFoundError: house_data.csv` | Run `python generate_data.py` to generate dataset |
| Streamlit port already in use | Use `streamlit run app.py --server.port 8502` |
| PowerShell script execution disabled | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `venv\Scripts\activate` not recognized | Use `.\venv\Scripts\Activate.ps1` in PowerShell |

---

## 🏗️ Project Architecture Quick Reference

### File Responsibilities

```mermaid
flowchart TB
    subgraph Scripts ["🐍 Python Modules"]
        gen["generate_data.py<br/>Create synthetic CSV"]
        pre["preprocess.py<br/>Clean · Encode · Scale"]
        trn["train.py<br/>Train + Save models"]
        evl["evaluate.py<br/>MAE · RMSE · R²"]
        mdl["model.py<br/>Load + Predict"]
        app["app.py<br/>Streamlit Web UI"]
    end

    subgraph Data ["📂 Data"]
        csv[("house_data.csv")]
    end

    subgraph Models ["💾 Artifacts"]
        lr[("Linear Regression<br/>model.joblib")]
        rf[("Random Forest<br/>model.joblib")]
    end

    gen --> csv
    csv --> pre
    pre --> trn
    evl --> trn
    trn --> lr
    trn --> rf
    lr --> mdl
    rf --> mdl
    mdl --> app

    style Scripts fill:#E8F5E9
    style Data fill:#E3F2FD
    style Models fill:#FFF3E0
```

### Quick Command Reference

| Action | Command |
|--------|---------|
| Generate data | `python generate_data.py` |
| Train all models | `python train.py` |
| Train specific model | `python train.py --model random_forest` |
| Launch app | `streamlit run app.py` |
| Deactivate venv | `deactivate` |

---

<div align="center">

### 🌟 Thank you for using & contributing to House Price Predictor!

**Questions?** Open an [issue](https://github.com/Subhadip-Paul2006/house_price_predictor/issues) on GitHub.

---

_USAGE.md v1.0 — Last updated: June 2026_

</div>
]]>
