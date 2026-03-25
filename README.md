# Tree-Based-Classifiers-Boolean-Datasets
This project compares four tree-based classifiers on 15 Boolean classification datasets:

- Decision Tree
- Bagging
- Random Forest
- AdaBoost

The program tunes each model using the validation set, retrains the best version on train + validation, and evaluates final performance on the test set.

## Files

- `main.py` – runs all experiments
- `all_data/` – contains the train, validation, and test CSV files
- `results/` – saves tuning grids, summary CSVs, and final comparison tables
- `ML_Project2_Report.pdf` – final written report

## How to Run

1. Make sure `main.py` and the `all_data` folder are in the same project directory.
2. Install required libraries if needed:
   ```bash
   pip install pandas scikit-learn
   
Run:

```bash
python main.py
