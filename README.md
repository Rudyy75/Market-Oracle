# Market Oracle v2 🔮

> **ML-powered stock market direction prediction with LSTM, Attention, and Sentiment Analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📊 Project Overview

Market Oracle is a comprehensive ML project that predicts stock market direction using:
- **Phase 1:** Classical ML (Logistic Regression, Random Forest)
- **Phase 2:** Deep Learning (LSTM networks)
- **Phase 3:** NLP + Attention (Sentiment analysis + Attention mechanism)
- **Phase 4:** Production (Backtesting, CI/CD, Docker)

---

## 🏗️ Project Structure

```
Market-Oracle/
├── config/
│   └── config.yaml          # All hyperparameters
├── data/
│   ├── raw/                  # Downloaded OHLCV data
│   └── processed/            # Feature-engineered data
├── models/
│   └── phase1/               # Saved model weights
├── notebooks/                # Jupyter/Colab notebooks
├── tests/                    # pytest test suite
├── outputs/
│   ├── figures/              # Generated charts
│   └── results/              # Metrics CSVs
├── utils/                    # Helper modules
├── docs/                     # Documentation
├── data_loader.py            # Data fetching
├── indicators.py             # Technical indicators
├── sentiment.py              # News sentiment
├── windowing.py              # Time-series windows
├── walk_forward.py           # Validation splits
├── backtester.py             # Trading simulation
├── visualization.py          # Chart generation
├── main.py                   # CLI pipeline
├── requirements.txt          # Local dev deps
└── requirements-colab.txt    # Colab training deps
```

---

## 🚀 Quick Start

### Local Development
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/Market-Oracle.git
cd Market-Oracle

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

### Google Colab (GPU Training)
```python
# Mount Drive and clone repo
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/YOUR_USERNAME/Market-Oracle.git
%cd Market-Oracle
!pip install -r requirements-colab.txt
```

---

## 📅 Development Timeline

| Phase | Focus | Duration |
|-------|-------|----------|
| Phase 1 | ML Foundations + Classifiers | Week 1 |
| Phase 2 | LSTM + Deep Learning | Week 2 |
| Phase 3 | NLP + Sentiment + Attention | Week 3 |
| Phase 4 | Backtesting + Production | Week 4 |

---

## 📈 Results

*Coming soon after Phase 1 completion*

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.
