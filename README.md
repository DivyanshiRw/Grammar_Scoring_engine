# Spoken English Grammar Scoring Engine

An automated scoring engine designed to predict continuous MOS Likert grammar scores (1.0 to 5.0) from spoken audio samples (45–60 seconds). Built with a dual-branch pipeline combining automatic speech recognition (ASR), dense text embeddings, and acoustic fluency analysis.

---

## 1. Project Overview

- **Input:** Standard `.wav` audio files (45–60s duration).
- **Output:** Continuous grammar proficiency score bounded within `[1.0, 5.0]`.
- **Dataset Size:**
  - Training set: 769 audio instances with human MOS labels.
  - Evaluation set: 216 test instances.
- **Primary Metrics:** Root Mean Squared Error (RMSE) and Pearson Correlation Coefficient ($r$).

---

## 2. Pipeline Architecture

Given the sample size (769 training instances) and CPU execution constraints, training an end-to-end deep neural network directly on raw audio introduces severe overfitting risk. The pipeline separates evaluation into complementary linguistic, acoustic, and machine learning components:

```text
Audio (.wav) ───┬──► Whisper-tiny ASR ──► all-mpnet-base-v2 (768d) ───┐
                │                                                      │
                ├──► Text Extraction  ──► Syntax & Linguistic Metrics ─┼──► StandardScaler ──► RidgeCV + LightGBM Ensemble ──► MOS Score
                │                                                      │
                └──► Librosa Audio   ──► Acoustic Fluency Features ───┘
```

1. **Linguistic & Syntactic Representation:**
   - Raw audio transcribed using open-source openai/whisper-tiny.en.
   - Transcripts capture grammatical errors, false starts, and fragmented syntax.
   - Text converted into 768-dimensional dense semantic vectors using sentence-transformers/all-mpnet-base-v2.

2. **Explicit Grammatical & Syntactic Features:**

   - Lexical Richness: Type-Token Ratio (TTR) and average word length.
   - Syntactic Complexity: Clause density, sentence counts, and average sentence length.
   - Dysfluency Markers: Filler word frequency (um, uh, like) and consecutive word repetition rates (stutters/false starts).

3. **Acoustic Fluency Features:**
   - Extracted using librosa:
        - Total duration (seconds)
        - Total word count
        - Speech rate (words per second)
        - Silence / pause ratio (energy-based voice activity detection)

4. **Modeling & Regularization:**
   - Multi-modal features combined and scaled via StandardScaler.
   - 50/50 Ensemble: Blends L2-regularized Ridge Regression (RidgeCV) with a gradient-boosted tree model (LightGBM)
   - Evaluated across a 5-Fold Cross-Validation strategy.
   - Predictions clipped strictly to the rubric bounds $[1.0, 5.0]$.

---

## 3. Project Structure

```text
Grammar_Scoring_engine/
│
├── Dataset_Final/
│   ├── train/                 # Training .wav audio files
│   ├── test/                  # Test .wav audio files
│   ├── train.csv              # Filenames and ground truth MOS labels
│   ├── test.csv               # 216 test filenames
│   └── sample_submission.csv  # Formatting reference
│
├── 01_feature_extraction.ipynb # Notebook 1: Audio transcription & feature caching
├── 02_model_and_report.ipynb   # Notebook 2: Initial baseline model & report
├── 03_experiment_lgbm.ipynb    # Notebook 3: SOTA MPNet + LightGBM + Ridge ensemble
├── train_features.csv         # Cached training features (ASR + acoustic)
├── test_features.csv          # Cached test features (ASR + acoustic)
├── evaluation_plots.png       # Regression fit and residual distribution plots
├── submission.csv             # Final test predictions (216 rows, score: 0.5407)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 4. Setup & Installation

```Bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (cmd/PowerShell):
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 5. Execution Workflow

**Step 1: Feature Extraction (`01_feature_extraction.ipynb`)**
- Configures `imageio-ffmpeg` for Whisper on Windows/Linux.
- Generates speech transcripts and extracts acoustic fluency metrics via `librosa`.
- Caches representations to `train_features.csv` and `test_features.csv`.

**Step 2: Training, Evaluation & Submission (`03_experiment_lgbm.ipynb`)**
- Computes sentence embeddings using `all-mpnet-base-v2`.
- Extracts linguistic complexity and stutter/filler metrics.
- Performs 5-Fold Cross-Validation across RidgeCV and LightGBM models.
- Generates diagnostic visualizations and final ensembled `submission.csv` for the 216 test instances.

---

## 6. Evaluation Results

### Cross-Validation & Progression Benchmarks

| Iteration / Experiment Architecture | Train RMSE | 5-Fold CV RMSE | 5-Fold Pearson ($r$) | Public LB Score |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline:** MiniLM-L6 + Acoustic (RidgeCV) | 0.6946 | 0.8531 | 0.7288 | 0.6235 |
| **Iteration 2:** MiniLM-L6 + Syntax + LightGBM/Ridge Blend | 0.4971 | 0.7901 | 0.7849 | 0.5694 |
| **Final SOTA:** MPNet-base + Syntax + LightGBM/Ridge Blend | **0.4612** | **0.7433** | **0.8125** | **0.5407** |

### Key Observations
* **High Linear Correlation ($r = 0.8125$):** Surpassing 0.81 confirms strong ordinal and relative ranking consistency matching human raters.
* **Low Spread of Residuals:** Residual errors follow a zero-centered Gaussian distribution without systematic positive or negative grading bias.
* **Resilience to Overfitting:** The modest gap between Training RMSE (0.4612) and Validation RMSE (0.7433) demonstrates that L2 regularization and conservative tree depth prevent memorization on small datasets.


