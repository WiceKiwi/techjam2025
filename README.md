# TechJam2025 Review Quality Project

## Overview
This project builds a full pipeline for assessing the quality and genuineness of Google location reviews.  
The system detects spam, ads, irrelevant reviews, and rants, while also scoring relevancy and visit likelihood.  
We leverage **rule-based heuristics**, **silver labeling with LLMs**, and **traditional ML training** to produce reliable classifiers for policy enforcement.

---
## Prerequisites
### Setup Environment
```bash
conda create --name ratu python=3.12
conda activate ratu
pip install -r requirements.txt
```

### Input Data
Due to time constraints, we will be using a smaller dataset limited to the Alaska region, available at https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal. At the root of the project directory, create the folder:
```
datasets/<thedataset> # json.gz
datasets/<meta> #json.gz
```

We will use both the metadata for Alaska and the Alaska review subset.

## Pipelines

### 1. DataCollectionPipeline
- Collects and preprocesses raw review data.
- Adds metadata features (length, emojis, rating deviation, etc.).
- Ensures `review_id` is generated consistently.

### 2. FeatureEngineeringPipeline
- Applies handcrafted rules (regex, heuristics) to flag spam, ads, irrelevant, and rant reviews.

### 3. RuleBasedPipeline
- Produces rule scores and binary strong indicators.

### 4. SilverLabelingPipeline
- Uses an LLM (Gemma-3-12B-it or similar) to produce "silver" probabilistic labels for each review.
- Extracts structured JSON scores for downstream training.

### 5. GoldLabelingPipeline
- (Optional, when human annotations are available)
- Stratified sampling of reviews for annotation.
- Merges human-provided gold labels back into dataset.

### 6. DatasetSplitPipeline
- Cleans dataset, applies hygiene filters.
- Stratified split into **train / validation / test** with class balance guarantees.
- Outputs per-split JSONL and fold CSV for CV.

### 7. ModelTrainingPipeline
- Trains **LightGBM models** for classification (ads_promo, spam_low_quality, irrelevant, rant_no_visit)  
  and regression (relevancy_score, visit_likelihood).
- Saves models, tuned thresholds, and validation/test predictions.

### 8. EvaluationPipeline
- Evaluates predictions on val/test splits.
- Computes metrics (precision, recall, F1, ROC-AUC, PR-AUC, calibration, etc.).
- Generates plots and Markdown summary reports.

### 9. PolicyEnforcementPipeline
- Combines model outputs into final **policy flags**.
- Flags reviews as ads, spam, irrelevant, rant, or genuine.
- Produces final dataset for decision-making.

---

## How to Run

| Pipeline                  | Script Command Example |
|---------------------------|-------------------------|
| Data Collection           | `python -m scripts.run_data_collection --config configs/data_collection.yaml` |
| Feature Engineering           | `python -m scripts.run_feature_engineering --config configs/feature_engineering.yaml` |
| Rule-based Scoring        | `python -m scripts.run_rules_baseline --config configs/rules_baseline.yaml` |
| Silver Labeling           | `python -m scripts.run_silver_labeling --config configs/silver_labeling.yaml` |
| Gold Labeling (optional)  | `python -m scripts.run_gold_labeling --config configs/gold_labeling.yaml` |
| Dataset Split             | `python -m scripts.run_dataset_split --config configs/dataset_split.yaml` |
| Model Training            | `python -m scripts.run_model_training --config configs/model_training.yaml` |
| Evaluation                | `python -m scripts.run_evaluation --config configs/evaluation.yaml` |
| Policy Enforcement        | `python -m scripts.run_policy_enforcement --config configs/policy_enforcement.yaml` |

---

## Demo Script
run:

```bash
python -m scripts.demo --config configs/demo.yaml
```

The demo will:
1. Load a small sample of reviews.
2. Run rule-based checks + silver labeling.
3. Train lightweight models quickly.
4. Show evaluation summary and final policy flags.

This demonstrates the end-to-end flow without requiring the full dataset.

For simple UI demonstration of our plicy enforcement, you can run:
```bash
streamlit run streamlit_demo.py
```
