# Market Oracle v2 - ML Project

# **THE 4-WEEK DAY-BY-DAY ROADMAP**

---

# **🟦 WEEK 1 — ML FOUNDATIONS + PHASE 1 CLASSIFIER**

**Goal:**

➡ Learn ML fundamentals

➡ Build feature engineering pipeline

➡ Build a walk-forward validated logistic regression & random forest classifier

---

## **DAY 1 — Python + pandas Refresher (2–3 hrs)**

### Learn:

- pandas indexing
- rolling windows
- merging
- matplotlib basics

### Resources:

- Corey Schafer pandas playlist (best intro)
- aiml.com → “Python for ML” basics

### Code Tasks:

- Load a CSV in pandas
- Plot close price
- Compute log returns manually

---

## **DAY 2 — ML Fundamentals (4 hrs)**

### Learn:

- What is supervised learning
- Classification vs regression
- Overfitting
- Train/test splits
- Walk-forward validation concept

### Resources:

- aiml.com → ML Fundamentals
- D2L Section 1.1 + 2.1

### Code Tasks:

- Implement simple train/test split
- Train logistic regression using scikit-learn

---

## **DAY 3 — Downloading Stock Data + Cleaning (3 hrs)**

### Learn:

- Time-series indexing
- Handling NA
- Visualizing trends

### Resources:

- yfinance documentation

### Code Tasks:

- Build a script `data_loader.py`:
    - Download ticker data
    - Save CSV to `/data/raw/`
    - Clean NA values
    - Compute log returns

---

## **DAY 4 — Technical Indicators (3–4 hrs)**

### Learn:

- RSI
- MACD
- SMA
- Rolling volatility

### Resources:

- YouTube: “Compute RSI MACD in Python”
- aiml.com → Feature Engineering

### Code Tasks:

Implement in `indicators.py`:

- RSI
- MACD
- SMA 50
- SMA 200
- Rolling std

---

## **DAY 5 — Walk-Forward Validation (4 hrs)**

### Learn:

- Why random splits are invalid for time-series
- Expanding window approach

### Resources:

- “MachineLearningMastery Walk Forward Validation” article

### Code Tasks:

Implement `walk_forward.py`:

- Split data into folds
- For each fold: train on earlier data, test on next segment

---

## **DAY 6 — Phase 1 Model Training (4–5 hrs)**

### Learn:

- Logistic Regression details
- Random Forest (feature importance)

### Code Tasks:

Notebook `03_phase1_classifier.ipynb`:

- Train both models
- Evaluate accuracy, precision
- Plot feature importance

---

## **DAY 7 — Review + Documentation Day**

### Tasks:

- Clean code
- Document feature pipeline
- Make README section for Phase 1
- Push Week 1 progress to GitHub

---

# 🟦 WEEK 2 — DEEP LEARNING FOUNDATIONS + LSTM MODEL (PHASE 2)

**Goal:**

➡ Learn deep learning basics

➡ Understand LSTM fully

➡ Build LSTM for log-return forecasting

➡ Use walk-forward again

---

## **DAY 8 — Neural Network Basics (3 hrs)**

### Learn:

- What are layers
- Weights, biases
- Activation functions
- Loss functions
- Gradient descent

### Resources:

- aiml.com → Deep Learning Basics
- D2L 3.6 + 4.1

---

## **DAY 9 — RNNs & LSTMs Theory (4 hrs)**

### Learn:

- Why RNNs fail (vanishing gradient)
- How LSTM solves it
- Forget gate
- Cell state

### Resources:

- D2L 8.1, 8.2, 9.1
- StatQuest LSTM video

---

## **DAY 10 — TensorFlow Fundamentals (3 hrs)**

### Learn:

- Keras Sequential API
- Layers
- Loss + optimizer
- Training loop

### Resource:

- TensorFlow Beginner Tutorials

### Code Tasks:

- Make a toy neural network (MNIST)
- Train for 5 epochs

---

## **DAY 11 — Windowing Time Series (3 hrs)**

### Learn:

- Sliding windows
- Supervised dataset creation

### Code Tasks:

Implement function:

```
create_windows(data, window=30)

```

Outputs:

- X : (samples, 30, features)
- y : next-day log return

---

## **DAY 12 — Build LSTM Model (4–5 hrs)**

### Code Tasks:

Notebook `04_lstm_model.ipynb`:

- LSTM(64 units)
- Dropout
- Dense(1)
- Loss: MSE
- Optimizer: Adam

---

## **DAY 13 — Walk-Forward Training + Metrics (3 hrs)**

### Code Tasks:

- Apply walk-forward to LSTM
- Save RMSE results
- Compute “direction accuracy” (sign match)

### Visualization:

- Plot predicted vs actual returns

---

## **DAY 14 — Review + Refactor**

- Clean code
- Add comments
- Update README
- Push to GitHub

---

# 🟦 WEEK 3 — NLP + SENTIMENT + ATTENTION (PHASE 3)

**Goal:**

➡ Learn sentiment analysis

➡ Integrate sentiment with price features

➡ Understand attention

➡ Build multi-input LSTM+Attention model

---

## **DAY 15 — NLP Basics (3 hrs)**

### Learn:

- Tokenization
- Stopwords
- Bag of words
- Why simple sentiment models work

### Resources:

- aiml.com NLP Intro
- D2L NLP Intro

---

## **DAY 16 — VADER Sentiment (2 hrs) + News Scraping (2 hrs)**

### Learn:

- Using NLTK VADER
- Aggregating daily sentiment
- Avoiding leakage: use sentiment(t) → predict(t+1)

### Code Tasks:

`sentiment.py`:

- Fetch news (NewsAPI)
- Compute VADER score
- Merge with price dataset

---

## **DAY 17 — Attention Mechanism Theory (3 hrs)**

### Learn:

- Why attention helps
- Key / value / query concept

### Resources:

- D2L 9.4 Attention
- YouTube: “Attention Explained Simply”

---

## **DAY 18 — Implement Attention Layer (3–4 hrs)**

### Code Tasks:

Custom Keras layer:

```
score = tanh(W1*h + b1)
attention_weights = softmax(score)
context = sum(attention_weights * h)

```

Add before Dense layer.

---

## **DAY 19 — Train Attention LSTM Model (4 hrs)**

### Code Tasks:

`05_attention_model.ipynb`:

- Train walk-forward
- Compare RMSE
- Compare direction accuracy
- Plot attention weights

---

## **DAY 20 — Evaluate Sentiment Impact (3 hrs)**

### Analysis Tasks:

- Compare LSTM vs LSTM+Sentiment
- Compare LSTM vs Attention model
- Save results table

---

## **DAY 21 — Documentation Day**

- Clean everything
- Update README Phase 3 section
- Push Week 3 to GitHub

---

# 🟦 WEEK 4 — BACKTESTING ENGINE + FINALIZATION

**Goal:**

➡ Build a simple trading engine

➡ Evaluate Sharpe, drawdown, CAGR

➡ Produce charts

➡ Finalize GitHub + Report

---

## **DAY 22 — Backtesting Basics (3 hrs)**

### Learn:

- Buy/sell strategy
- Drawdown calculation
- Sharpe ratio

### Resource:

- QuantInsti YouTube basics on backtesting

---

## **DAY 23 — Implement Backtesting Engine (4 hrs)**

### Code Tasks:

`backtester.py`:

Inputs: predicted returns

Rules:

```
IF predicted return > 0 → BUY
ELSE → STAY IN CASH

```

Outputs:

- Equity curve
- CAGR
- Sharpe (mean/vol)
- Max drawdown

---

## **DAY 24 — Full Pipeline Integration (4 hrs)**

Script `main.py`:

- Load raw data
- Compute indicators
- Add sentiment
- Create windows
- Run model
- Backtest strategy
- Save results

---

## **DAY 25 — Visualization + Metrics Summary (3 hrs)**

Charts to generate:

- Equity curve
- Daily returns
- Attention heatmaps
- Feature importance (Phase 1)

---

## **DAY 26 — Write Final Report (4 hrs)**

Sections:

1. Abstract
2. Problem Statement
3. Data
4. Methodology
5. Models
6. Results
7. Backtest performance
8. Limitations
9. Future Work

---

## **DAY 27 — Final GitHub Polish**

README MUST INCLUDE:

- Architecture diagram
- Results summary
- Installation instructions
- Usage guide

---

## **DAY 28 — Buffer Day (Fix bugs + Viva prep)**

Prepare answers for:

- Why walk-forward validation?
- Why not predict price?
- Why LSTM?
- Why attention?
- How did sentiment help?

---

# 🟩 **OUTPUT BY END OF 4 WEEKS**

You will have:

✔ Fully working ML + DL + NLP + Quant project

✔ Professional GitHub repo

✔ Research-style PDF report

✔ Beautiful plots

✔ Production-ready code structure

✔ Resume-grade achievement

This is **strong enough to be your capstone project or internship centerpiece**.