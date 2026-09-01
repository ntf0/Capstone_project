# Personalized Learning Through Student Behavior Clustering

**Author:** Elias Abdulshaheed A.ameer Abbas
**Program:** Data Science Bootcamp · Cohort DSB-2 Full Time · Capstone Project

A behavior-driven clustering pipeline that turns raw ASSISTments student interaction logs into four named, actionable learning personas — plus a working Streamlit prototype that demonstrates how those personas could power real-time, AI-personalized practice assignments.

---

## Overview

Online tutoring platforms like [ASSISTments](https://new.assistments.org/) capture far more than right-or-wrong answers: how many attempts a student needed, how many hints they asked for, how long they took to respond, and how much prior practice they'd had on a skill. Most platforms still treat every student on a given skill the same way.

This project takes an **unsupervised learning approach** instead — clustering students by *how* they engage with problems, not just whether they got them right — and turns those clusters into interpretable personas educators can act on.

**Problem statement:** Educators struggle to personalize support at scale. This project groups students into behavioral clusters with the aim of improving student performance by 10% through better-targeted, personalized support.

## Data

- **Source:** ASSISTments Skill Builder dataset, 2009–10 school year
- **346,860** raw student–problem interaction records
- **101** distinct math skills covered (e.g. equation solving, fraction conversion)
- Aggregated by student × skill into **34,690** behavior profiles, **34,438** retained after outlier filtering
- **7** final engineered behavioral features (down from 12 candidate signals), after removing redundant/duplicate signals and power-transforming heavily skewed ones

**Key engineered features:** `accuracy_rate`, `avg_attempts`, `retry_rate`, `avg_hint_dependency`, `median_response_time`, `max_opportunity`, `total_interactions`

## Methodology

1. **Data cleaning** — corrected inconsistent performance records, handled missing `bottom_hint` values, removed implausibly short response times and zero-attempt records, fixed inconsistent `skill_id`s.
2. **Feature engineering & preprocessing** — aggregated raw logs to student × skill level, dropped highly correlated/duplicate features (e.g. `first_try_success_rate` mirrored `accuracy_rate` at r ≈ 1.00), applied Box-Cox/Yeo-Johnson power transforms and scaling (Standard/Robust) via a single scikit-learn `ColumnTransformer`.
3. **Modeling** — trained and compared three clustering approaches, tuning hyperparameters (k, init method) with grid search:
   - K-Means
   - K-Means + PCA (3 components, retaining 89.9% of variance)
   - Gaussian Mixture Model (GMM)
4. **Evaluation** — Silhouette Score (cluster separation, higher is better) and Davies–Bouldin Index (cluster compactness, lower is better).

| Model | Config | Silhouette ↑ | Davies–Bouldin ↓ |
|---|---|---|---|
| K-Means | k = 4 | 0.288 | 1.288 |
| K-Means + PCA | k = 4 | 0.317 | 1.112 |
| GMM | n = 2 | 0.300 | 1.300 |

**Selected model: plain K-Means, k = 4 (no PCA).** Not the top scorer numerically, but chosen because it produces centroids directly in the original feature space — clusters can be described with concrete, interpretable values (e.g. "76% accuracy, 76-second median response time") rather than abstract principal components. K-Means + PCA's higher metrics partly reflect PCA compressing the feature space rather than more real-world-distinct behavior, and GMM was ruled out on both metrics and structural fit.

## The Four Personas

| Persona | Share | Profile | Implication |
|---|---|---|---|
| **Active Strivers** | 28.2% | 56% accuracy, highest practice volume | Need targeted practice and feedback, not just more repetition |
| **Independent Solvers** (Deliberate Masters) | 28.0% | 100% accuracy, lowest retries & hints | Ready for enrichment or more challenging material |
| **Overwhelmed & Struggling** | 14.2% | 27% accuracy, highest hint & retry use | Highest priority for scaffolding and teacher support |
| **Efficient Practicers** | 29.6% | 88% accuracy, fastest responses | Ready to progress with less support |

## Limitations

- **Static snapshot** — fixed behavioral aggregates; doesn't capture how behavior changes over time, or skill difficulty/prior knowledge.
- **Modest separation** — Silhouette and Davies–Bouldin scores were modest across all three models; some students sit near the boundary between personas.
- **Sparse-data risk** — the Independent Solvers/Deliberate Masters cluster is based on relatively few interactions per student, so it may partly reflect sparse data rather than a stable pattern.
- **Behavior, not cause** — these are unsupervised, purely behavioral clusters. They describe what students did, not why, and shouldn't be read as judgments of ability or motivation.

## Future Improvements

1. **AI-generated personalized questioning** — pass a student's cluster assignment and behavior description to an LLM so question difficulty and scaffolding adapt automatically.
2. **IoT-based data capture** — enrich interaction data collection beyond the platform log.
3. **Longitudinal re-clustering** — periodically recompute cluster membership as new data accumulates.
4. **National-scale integration** — partner with education ministries to deploy behavior-informed support across schools, with added validation and privacy safeguards.

## Prototype: AI-Powered Personalized Learning App

To demonstrate the pipeline end-to-end, this repo includes a Streamlit prototype (`app.py`) that:

1. Runs a student through a fixed 5-question diagnostic assignment (Fractions & Algebraic Expressions), tracking accuracy, attempts, hints used, and response time per question.
2. Feeds that session's behavioral log into the saved clustering pipeline (`student_clustering_pipeline.joblib`) to predict the student's persona in real time.
3. Sends the persona and behavioral features to the Gemini API, which generates 3 new, personalized follow-up questions targeting the same skill — harder and less scaffolded for high performers, more scaffolded and supportive for struggling students.
4. Logs each completed session (accuracy, avg. attempts, persona, generated assignment) to a local SQLite database (`student_history.db`).

### Running the prototype

```bash
pip install streamlit google-genai joblib scikit-learn pandas numpy
setx GEMINI_API_KEY "your-key-here"      # Windows (open a new terminal after)
export GEMINI_API_KEY="your-key-here"    # Mac/Linux

streamlit run app.py
```

## Project Structure

```
.
├── Final_Notebook.ipynb                          # Full analysis: EDA, cleaning, feature engineering, modeling, evaluation
├── app.py                                        # Streamlit prototype: diagnostic quiz → persona prediction → AI-generated follow-up
├── pipeline_utils.py                             # Shared feature-building & persona label/description helpers
├── student_clustering_pipeline.joblib            # Trained K-Means (k=4) preprocessing + clustering pipeline
├── student_history.db                            # SQLite log of completed prototype sessions (created on first run)
└── Student_Behavior_Clustering_Capstone_FinalV.pptx   # Capstone presentation deck
```

## Tech Stack

- **Analysis:** Python, pandas, NumPy, scikit-learn (K-Means, GMM, PCA, `ColumnTransformer`, `PowerTransformer`), SciPy, Matplotlib, Seaborn
- **Prototype app:** Streamlit, joblib, Google Gemini API (`google-genai`), SQLite

## Acknowledgments

With thanks to the Data Science Bootcamp instructors and program leadership — Ahmed Fakhr, Husain Amer, Jubran Jalal, Fatema Algallaf, Najim Alfutini, Esraa Haji, and Ahlam Oun — for their guidance and support throughout this capstone.
