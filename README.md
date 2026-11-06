# Smart Notes Assistant 🧠

> **"Your thoughts, intelligently organized."**

An intelligent, machine-learning-powered note-taking and retrieval system that automatically understands, categorizes, organizes, and retrieves everyday notes using Natural Language Processing (NLP) with **TF-IDF vectorization**, **Logistic Regression**, and **uncertainty detection**.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Key Features](#key-features)
4. [Technology Stack](#technology-stack)
5. [Application Architecture](#application-architecture)
6. [Machine Learning Workflow Explained](#machine-learning-workflow-explained)
7. [Step-by-Step Setup on Mac using VS Code](#step-by-step-setup-on-mac-using-vs-code)
8. [How to Train the Model](#how-to-train-the-model)
9. [How to Run the Application](#how-to-run-the-application)
10. [Running Automated Tests](#running-automated-tests)
11. [Intelligent Search & Benchmark](#intelligent-search--benchmark)
12. [Known Limitations & Future Enhancements](#known-limitations--future-enhancements)

---

## 1. Project Overview

Every day, people write down thoughts, to-dos, bills, flight details, medical appointments, and study tasks in natural language. Traditional note apps force users to manually click through folders, pick tags, or rely solely on exact keyword searches.

**Smart Notes Assistant** bridges this gap: as soon as you enter a note like:
- *"Flight to Delhi is booked for Monday morning."* → Automatically classified as **Travel** (93% confidence)
- *"Pay the electricity bill before the 28th."* → Automatically classified as **Finance** (96% confidence)
- *"My Python exam is on Friday at 10 AM."* → Automatically classified as **Education** (94% confidence)

If a note is vague or ambiguous (e.g., *"Call grandma this weekend."*), the assistant detects the low prediction margin and flags an **Uncertainty Alert**, inviting you to confirm or pick the category manually.

---

## 2. Problem Statement

Manual folder and tag management creates friction that leads to disorganized, lost notes. Exact keyword search also fails when query terminology differs from the saved text (e.g., searching *"electricity payment"* fails to find a note containing *"electricity bill"* in basic keyword systems).

**Smart Notes Assistant** solves both problems:
1. **Automated ML Categorization**: Eliminates manual filing friction.
2. **Uncertainty Detection**: Never misplaces ambiguous notes silently; alerts the user with human-in-the-loop validation.
3. **Dual Search Engine**: Combines literal token keyword search with geometric **TF-IDF Cosine Similarity Search** for semantic intent matching.

---

## 3. Key Features

- **Automatic Category Prediction**: Predicts one of 8 sensible categories:
  - `Education`, `Work`, `Personal`, `Finance`, `Travel`, `Shopping`, `Health`, `Events`.
- **Uncertainty Detection**: Triggers when model confidence drops below the threshold or the prediction margin between the top two categories is too narrow.
- **Manual Category Correction**: Users can easily override the predicted category, updating the note's confidence score to 100%.
- **Persistent Storage**: SQLite database (`data/notes.db`) persists notes across app restarts with JSON export and import capabilities.
- **Smart Note Management**: Full CRUD (Create, Read, Update, Delete) with creation dates, statistics, category filters, and sample note loading with strict duplicate prevention.
- **Dual Search Engine**:
  - **Keyword Search**: Exact token matching, substring matching, and phrase bonuses.
  - **TF-IDF Similarity Search**: Vector dot product with cosine similarity for natural language retrieval.
- **Retrieval Benchmark**: Side-by-side evaluation using **Hit Rate @ 1**, **Hit Rate @ 3**, and **Mean Reciprocal Rank (MRR)**.
- **Analytics & Model Evaluation**: Live test-set metrics including Accuracy, Precision, Recall, Macro-F1, and Confusion Matrix.

---

## 4. Technology Stack

- **Programming Language**: Python 3.10+
- **User Interface**: Streamlit
- **Machine Learning**: Scikit-learn (`TfidfVectorizer`, `LogisticRegression`)
- **Data & Linear Algebra**: Pandas, NumPy
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) with sublinear scaling
- **Storage**: SQLite (`sqlite3`) with JSON export
- **Testing**: Pytest / Python `unittest`

---

## 5. Application Architecture

```
smart-notes-assistant/
│
├── app.py                  # Main Streamlit web application
├── classifier.py           # ML module: Preprocessing, TF-IDF, Logistic Regression, Uncertainty
├── train.py                # Training pipeline, stratified split, evaluation, model saving
├── search.py               # Keyword & TF-IDF Cosine Similarity search engines & benchmarks
├── storage.py              # SQLite storage layer with CRUD & duplicate prevention
├── requirements.txt        # Lightweight Python dependency definitions
├── run_tests.py            # Standalone zero-dependency test runner
│
├── data/
│   ├── training_notes.csv  # 200+ labelled notes across all 8 categories
│   ├── model.pkl           # Serialized trained classifier model
│   └── notes.db            # Persistent SQLite database file
│
└── tests/
    └── test_notes.py       # 12 automated unit and integration tests
```

---

## 6. Machine Learning Workflow Explained

### A. Preprocessing
1. **Case Normalization**: Converts all text to lowercase so `"Python"` matches `"python"`.
2. **Punctuation Stripping**: Removes punctuation and special symbols (`@`, `#`, `!`, `,`, `.`).
3. **Stop Word Removal**: Eliminates common grammatical fillers (`the`, `is`, `at`, `on`, `for`) that carry no category signal.

### B. TF-IDF Feature Extraction
- **Term Frequency ($TF$)**: Measures how often a word appears in a note, scaled sublinearly ($1 + \ln(TF)$).
- **Inverse Document Frequency ($IDF$)**: Penalizes words that appear across all notes and rewards words specific to a single domain (e.g., `"exam"` or `"flight"`).
- **L2 Normalization**: Scales note vectors to unit length so note length does not distort similarity calculations.

### C. Logistic Regression Classification
- Calculates a weighted score for each category:
  $$\text{score}_c = \text{bias}_c + \sum (w_{c,i} \times \text{tfidf}_i)$$
- Applies the **Softmax function** to transform raw scores into a probability distribution summing to 1.0 (100%).

### D. Uncertainty Detection
A note is marked uncertain (`is_uncertain = True`) if:
1. Top confidence is below `0.35` (35%), OR
2. Margin between the #1 and #2 candidate categories is under `0.12` (12%), OR
3. Note text lacks sufficient domain keywords.
In such cases, the UI displays:
> *"Not sure about this note's category. Please select a category manually."*

---

## 7. Step-by-Step Setup on Mac using VS Code

Follow these beginner-friendly instructions to run the application on your Mac:

### Step 1: Open VS Code and the Project Folder
1. Launch **Visual Studio Code**.
2. Go to **File → Open Folder...** and select this project directory.
3. Open the built-in terminal by pressing ``Ctrl + ` `` (Control + Backtick) or selecting **Terminal → New Terminal** from the top menu.

### Step 2: Create and Activate a Python Virtual Environment
Run the following in the VS Code terminal:

```bash
# Create a virtual environment named .venv
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```
*(Your terminal prompt will now show `(.venv)` at the beginning of the line).*

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 8. How to Train the Model

To train the TF-IDF feature representation and Logistic Regression classifier on the labelled dataset:

```bash
python3 train.py
```

**Expected Output:**
```
============================================================
 Smart Notes Assistant — ML Training Pipeline
============================================================
[*] Loading training dataset from: data/training_notes.csv
    Loaded 209 samples across 8 categories.
[*] Performing stratified 80/20 train/test split...
    Training samples: 169
    Testing samples:  40
[*] Training TF-IDF Vectorizer + Logistic Regression Classifier...
    Training completed successfully.
[*] Evaluating model on held-out test set...
------------------------------------------------------------
 Model Accuracy:  ~70-80%
 Macro Precision: ~70-80%
 Macro Recall:    ~70-80%
 Macro F1 Score:  ~70-80%
------------------------------------------------------------
[*] Confusion Matrix (Rows = Actual, Columns = Predicted)
[*] Saving trained model to: data/model.pkl
============================================================
```

---

## 9. How to Run the Application

Start the Streamlit user interface by running:

```bash
streamlit run app.py
```

Streamlit will launch and print:
```
  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```
Open **http://localhost:8501** in your browser (Safari, Chrome, or Edge).

---

## 10. Running Automated Tests

Run the test suite to verify that all 12 core functional requirements pass:

Using standard **Pytest**:
```bash
pytest tests/test_notes.py -v
```

Or using the built-in test runner:
```bash
python3 run_tests.py
```

**Verified Test Scenarios:**
- ✅ `test_save_and_retrieve_note`: Persistent storage read/write
- ✅ `test_add_multiple_notes`: Adding multiple notes to SQLite
- ✅ `test_edit_note`: Modifying text and category
- ✅ `test_delete_note`: Removing notes by ID
- ✅ `test_category_filtering`: Filtering by Education, Finance, etc.
- ✅ `test_duplicate_prevention`: Loading sample notes safely without duplicates
- ✅ `test_model_training_and_evaluation`: Training & metrics calculation
- ✅ `test_category_predictions`: Accurate category inference
- ✅ `test_uncertainty_detection`: Low-margin / ambiguous note detection
- ✅ `test_manual_category_correction`: User overriding ML prediction
- ✅ `test_keyword_search`: Token-based search
- ✅ `test_similarity_search`: TF-IDF cosine similarity search

---

## 11. Intelligent Search & Benchmark

Smart Notes Assistant provides two distinct search algorithms:

| Feature | Keyword Search | TF-IDF Cosine Similarity Search |
| :--- | :--- | :--- |
| **Matching Style** | Literal token / substring | Geometric angle between TF-IDF vectors |
| **Best For** | Exact part numbers, names, IDs | Natural language queries & intent |
| **Example Query** | `"electricity"` matches note | `"electricity payment"` finds `"electricity bill"` |
| **Scoring** | Token overlap fraction | Cosine similarity ($0.0 \rightarrow 1.0$) |

You can click **"Run Retrieval Benchmark Evaluation"** in the **Smart Search** tab to compare their **Hit Rate @ 3** side by side.

---

## 12. Known Limitations & Future Enhancements

1. **Short Notes with OOV (Out-of-Vocabulary) Words**: Notes with slang or rare brand names not present in the training set may trigger uncertainty. This is by design to ensure transparency.
2. **Grammar Independence**: Bag-of-words / TF-IDF does not preserve deep syntactic word order.
3. **Future Enhancements**:
   - Transformer embeddings (e.g., `all-MiniLM-L6-v2` with Sentence-Transformers) for semantic search.
   - Active learning: Retraining the classifier incrementally when users manually correct predictions.
   - Calendar / reminder exports (`.ics` format).
