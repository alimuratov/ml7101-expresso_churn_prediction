# Expresso Churn Prediction

ML7101 final project. We predict which Expresso customers will churn (leave the service).
Expresso is an African telecom company. The data comes from the [Zindi challenge](https://zindi.world/competitions/expresso-churn-prediction).
The metric is AUC.

## Team

| Member | Role |
| --- | --- |
| Lucky | Problem understanding and baselines |
| Ali | Feature engineering and feature selection |
| Soumedhik | Model optimization and final training |

## Repository

```
README.md
requirements.in / requirements.txt
Starter/
  Final_Notebook_Churn_Pred.ipynb   main notebook
  helpers.py                        pipeline, evaluation and plotting helpers
  VariableDefinitions.csv           column descriptions
  Older Notebook Runs/              earlier drafts
```

## Setup

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   pip install shap
   ```
2. Download `Train.csv`, `Test.csv` and `SampleSubmission.csv` from Zindi.
3. Put them in `Starter/Data/`.
4. Run `Starter/Final_Notebook_Churn_Pred.ipynb` from top to bottom.

## Notebook outline

1. **Problem and data.** `CHURN` is the target. We drop `MRG` because it has a single value. That leaves 3 categorical and 13 numerical features.
2. **Missing values.** The data is sparse, but we keep the missing values. They predict churn, since no data usually means no activity. We fill them with 0 or `"None"`. Zero-filled columns also get an `_IS_MISSING` flag, because 0 and missing can mean different things.
3. **Duplicates.** About 660k of 2.1M rows share identical features, and many have conflicting labels. We keep them as real noise. CV uses `GroupKFold`, grouped by a hash of the features, so duplicates never leak across folds.
4. **EDA.** Active customers churn less. Cramér's V shows the categorical signal comes mostly from missing values. Only about 19% of customers churn.
5. **Feature engineering.** We add `AVERAGE_TOP_UP_AMOUNT` and `INACTIVITY_COUNT` (columns that are missing, zero or `"None"`).
6. **Feature selection.** We compare ANOVA, L1 logistic regression, RFE with logistic regression and RFE with a decision tree. All four always keep `REGULARITY`, `ON_NET_IS_MISSING`, `INACTIVITY_COUNT` and `REGION`. ANOVA drops about half the features with almost no AUC loss, and it is fast, so we use it.
7. **Model comparison.** Logistic regression, random forest and gradient boosting on the same 5 grouped folds.
8. **Final model.** Histogram gradient boosting with class weighting, tuned by grid search, then refit on all training rows.
9. **Interpretation.** Permutation importance and SHAP.
10. **Submission.** Writes `hgb_final_submission.csv`.

A checkpoint cell saves and reloads results, so long steps do not need to be re-run.

## Results

| Model | CV AUC |
| --- | --- |
| Logistic regression | 0.9175 |
| Random forest | 0.9208 |
| Gradient boosting | 0.9211 |
| + class weighting | 0.9213 |
| + grid search (final) | **0.9219** |

- Random forest and boosting are tied. The gap is far smaller than the fold spread (about 0.012). We pick boosting because it is faster.
- Class weighting does not improve ranking. It moves the threshold. Recall rises from 0.61 to 0.91 and precision falls from 0.66 to 0.52. We keep it because a retention team wants to catch most churners.
- Best grid search setting: learning rate 0.1, 31 leaves, 300 samples per leaf.
- The final AUC is the grid search's out-of-fold score, since the refit uses all rows.
- All models land near 0.92. About 29% of customers have an identical record with the opposite label. No model on these features can separate them.

## Key findings

- `REGULARITY` (times active over 90 days) is the strongest feature. Shuffling it costs 0.14 AUC.
- `REGION` is second (0.04). Customers with no region churn about 45% of the time, against under 2% for the rest.
- Every other feature costs under 0.002 AUC.
- SHAP agrees. Low regularity pushes a customer toward churn.
- These results explain what the model uses, not why customers leave.
