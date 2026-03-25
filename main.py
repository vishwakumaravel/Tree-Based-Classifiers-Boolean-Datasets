import os
import re
import time
import warnings
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")


# =========================================================
# SETTINGS
# =========================================================

DATA_DIR = "all_data"
RESULTS_DIR = "results"

# Quick test mode:
# True  -> runs only first 1 dataset
# False -> runs all 15 datasets
QUICK_TEST = False

# Turn models on/off here
RUN_DECISION_TREE = True
RUN_BAGGING = True
RUN_RANDOM_FOREST = True
RUN_ADABOOST = True

RANDOM_STATE = 42


# =========================================================
# HELPERS
# =========================================================

def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def get_dataset_ids(data_dir):
    dataset_ids = []

    for filename in os.listdir(data_dir):
        match = re.match(r"train_(c\d+_d\d+)\.csv$", filename)
        if match:
            dataset_ids.append(match.group(1))

    def sort_key(name):
        match = re.match(r"c(\d+)_d(\d+)", name)
        c_val = int(match.group(1))
        d_val = int(match.group(2))
        return (c_val, d_val)

    dataset_ids.sort(key=sort_key)
    return dataset_ids


def load_split(data_dir, split_name, dataset_id):
    file_path = os.path.join(data_dir, f"{split_name}_{dataset_id}.csv")
    df = pd.read_csv(file_path, header=None)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


def combine_train_valid(X_train, y_train, X_valid, y_valid):
    X_full = pd.concat([X_train, X_valid], axis=0, ignore_index=True)
    y_full = pd.concat([y_train, y_valid], axis=0, ignore_index=True)
    return X_full, y_full


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return acc, f1


def safe_save_csv(df, path):
    df.to_csv(path, index=False)


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def make_bagging_classifier(base_tree, n_estimators):
    """
    Handles both newer and older sklearn versions.
    Newer uses estimator=...
    Older uses base_estimator=...
    """
    try:
        model = BaggingClassifier(
            estimator=base_tree,
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    except TypeError:
        model = BaggingClassifier(
            base_estimator=base_tree,
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    return model


def make_adaboost_classifier(base_tree, n_estimators):
    """
    Handles both newer and older sklearn versions.
    Newer uses estimator=...
    Older uses base_estimator=...
    """
    try:
        model = AdaBoostClassifier(
            estimator=base_tree,
            n_estimators=n_estimators,
            random_state=RANDOM_STATE
        )
    except TypeError:
        model = AdaBoostClassifier(
            base_estimator=base_tree,
            n_estimators=n_estimators,
            random_state=RANDOM_STATE
        )
    return model


# =========================================================
# DECISION TREE
# =========================================================

def tune_decision_tree(X_train, y_train, X_valid, y_valid):
    criteria = ["gini", "entropy"]
    max_depths = [5, 10, 20, None]
    min_splits = [2, 5, 10]
    min_leaves = [1, 2, 5]

    results = []
    best_params = None
    best_val_acc = -1

    for criterion in criteria:
        for max_depth in max_depths:
            for min_split in min_splits:
                for min_leaf in min_leaves:
                    model = DecisionTreeClassifier(
                        criterion=criterion,
                        max_depth=max_depth,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        random_state=RANDOM_STATE
                    )

                    model.fit(X_train, y_train)

                    train_pred = model.predict(X_train)
                    valid_pred = model.predict(X_valid)

                    train_acc, train_f1 = compute_metrics(y_train, train_pred)
                    valid_acc, valid_f1 = compute_metrics(y_valid, valid_pred)

                    results.append({
                        "criterion": criterion,
                        "max_depth": max_depth,
                        "min_samples_split": min_split,
                        "min_samples_leaf": min_leaf,
                        "train_accuracy": train_acc,
                        "train_weighted_f1": train_f1,
                        "valid_accuracy": valid_acc,
                        "valid_weighted_f1": valid_f1
                    })

                    if valid_acc > best_val_acc:
                        best_val_acc = valid_acc
                        best_params = {
                            "criterion": criterion,
                            "max_depth": max_depth,
                            "min_samples_split": min_split,
                            "min_samples_leaf": min_leaf
                        }

    return best_params, best_val_acc, pd.DataFrame(results)


def retrain_and_test_decision_tree(best_params, X_train, y_train, X_valid, y_valid, X_test, y_test):
    X_full, y_full = combine_train_valid(X_train, y_train, X_valid, y_valid)

    model = DecisionTreeClassifier(
        criterion=best_params["criterion"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=RANDOM_STATE
    )

    model.fit(X_full, y_full)
    test_pred = model.predict(X_test)
    test_acc, test_f1 = compute_metrics(y_test, test_pred)

    return test_acc, test_f1


# =========================================================
# BAGGING
# =========================================================

def tune_bagging(X_train, y_train, X_valid, y_valid):
    n_estimators_list = [10, 20, 50, 100]
    max_depth_list = [5, 10, None]

    results = []
    best_params = None
    best_val_acc = -1

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            base_tree = DecisionTreeClassifier(
                max_depth=max_depth,
                random_state=RANDOM_STATE
            )

            model = make_bagging_classifier(base_tree, n_estimators)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            valid_pred = model.predict(X_valid)

            train_acc, train_f1 = compute_metrics(y_train, train_pred)
            valid_acc, valid_f1 = compute_metrics(y_valid, valid_pred)

            results.append({
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "train_accuracy": train_acc,
                "train_weighted_f1": train_f1,
                "valid_accuracy": valid_acc,
                "valid_weighted_f1": valid_f1
            })

            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                best_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth
                }

    return best_params, best_val_acc, pd.DataFrame(results)


def retrain_and_test_bagging(best_params, X_train, y_train, X_valid, y_valid, X_test, y_test):
    X_full, y_full = combine_train_valid(X_train, y_train, X_valid, y_valid)

    base_tree = DecisionTreeClassifier(
        max_depth=best_params["max_depth"],
        random_state=RANDOM_STATE
    )

    model = make_bagging_classifier(base_tree, best_params["n_estimators"])
    model.fit(X_full, y_full)

    test_pred = model.predict(X_test)
    test_acc, test_f1 = compute_metrics(y_test, test_pred)

    return test_acc, test_f1


# =========================================================
# RANDOM FOREST
# =========================================================

def tune_random_forest(X_train, y_train, X_valid, y_valid):
    n_estimators_list = [10, 20, 50, 100]
    max_features_list = ["sqrt", "log2", 0.5]

    results = []
    best_params = None
    best_val_acc = -1

    for n_estimators in n_estimators_list:
        for max_features in max_features_list:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features=max_features,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            valid_pred = model.predict(X_valid)

            train_acc, train_f1 = compute_metrics(y_train, train_pred)
            valid_acc, valid_f1 = compute_metrics(y_valid, valid_pred)

            results.append({
                "n_estimators": n_estimators,
                "max_features": max_features,
                "train_accuracy": train_acc,
                "train_weighted_f1": train_f1,
                "valid_accuracy": valid_acc,
                "valid_weighted_f1": valid_f1
            })

            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                best_params = {
                    "n_estimators": n_estimators,
                    "max_features": max_features
                }

    return best_params, best_val_acc, pd.DataFrame(results)


def retrain_and_test_random_forest(best_params, X_train, y_train, X_valid, y_valid, X_test, y_test):
    X_full, y_full = combine_train_valid(X_train, y_train, X_valid, y_valid)

    model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_features=best_params["max_features"],
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_full, y_full)

    test_pred = model.predict(X_test)
    test_acc, test_f1 = compute_metrics(y_test, test_pred)

    return test_acc, test_f1


# =========================================================
# ADABOOST
# =========================================================

def tune_adaboost(X_train, y_train, X_valid, y_valid):
    n_estimators_list = [10, 20, 50, 100]
    base_depth_list = [1, 3, 5]

    results = []
    best_params = None
    best_val_acc = -1

    for n_estimators in n_estimators_list:
        for base_depth in base_depth_list:
            base_tree = DecisionTreeClassifier(
                max_depth=base_depth,
                random_state=RANDOM_STATE
            )

            model = make_adaboost_classifier(base_tree, n_estimators)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            valid_pred = model.predict(X_valid)

            train_acc, train_f1 = compute_metrics(y_train, train_pred)
            valid_acc, valid_f1 = compute_metrics(y_valid, valid_pred)

            results.append({
                "n_estimators": n_estimators,
                "base_depth": base_depth,
                "train_accuracy": train_acc,
                "train_weighted_f1": train_f1,
                "valid_accuracy": valid_acc,
                "valid_weighted_f1": valid_f1
            })

            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                best_params = {
                    "n_estimators": n_estimators,
                    "base_depth": base_depth
                }

    return best_params, best_val_acc, pd.DataFrame(results)


def retrain_and_test_adaboost(best_params, X_train, y_train, X_valid, y_valid, X_test, y_test):
    X_full, y_full = combine_train_valid(X_train, y_train, X_valid, y_valid)

    base_tree = DecisionTreeClassifier(
        max_depth=best_params["base_depth"],
        random_state=RANDOM_STATE
    )

    model = make_adaboost_classifier(base_tree, best_params["n_estimators"])
    model.fit(X_full, y_full)

    test_pred = model.predict(X_test)
    test_acc, test_f1 = compute_metrics(y_test, test_pred)

    return test_acc, test_f1


# =========================================================
# MAIN
# =========================================================

def main():
    start_time = time.time()
    ensure_results_dir()

    dataset_ids = get_dataset_ids(DATA_DIR)

    if QUICK_TEST:
        dataset_ids = dataset_ids[:1]

    print_header("DATASETS TO RUN")
    for ds in dataset_ids:
        print(ds)

    dt_summary_rows = []
    bag_summary_rows = []
    rf_summary_rows = []
    ada_summary_rows = []

    for dataset_id in dataset_ids:
        print_header(f"DATASET: {dataset_id}")

        X_train, y_train = load_split(DATA_DIR, "train", dataset_id)
        X_valid, y_valid = load_split(DATA_DIR, "valid", dataset_id)
        X_test, y_test = load_split(DATA_DIR, "test", dataset_id)

        # -------------------------
        # Decision Tree
        # -------------------------
        if RUN_DECISION_TREE:
            print("\nRunning Decision Tree...")

            best_params, best_val_acc, grid_df = tune_decision_tree(X_train, y_train, X_valid, y_valid)
            grid_path = os.path.join(RESULTS_DIR, f"decision_tree_grid_{dataset_id}.csv")
            safe_save_csv(grid_df, grid_path)

            test_acc, test_f1 = retrain_and_test_decision_tree(
                best_params, X_train, y_train, X_valid, y_valid, X_test, y_test
            )

            dt_summary_rows.append({
                "dataset": dataset_id,
                "criterion": best_params["criterion"],
                "max_depth": best_params["max_depth"],
                "min_samples_split": best_params["min_samples_split"],
                "min_samples_leaf": best_params["min_samples_leaf"],
                "best_validation_accuracy": best_val_acc,
                "test_accuracy": test_acc,
                "test_weighted_f1": test_f1
            })

            print("Best DT params:", best_params)
            print(f"DT validation accuracy: {best_val_acc:.4f}")
            print(f"DT test accuracy: {test_acc:.4f}")
            print(f"DT test weighted F1: {test_f1:.4f}")

            safe_save_csv(pd.DataFrame(dt_summary_rows), os.path.join(RESULTS_DIR, "decision_tree_summary.csv"))

        # -------------------------
        # Bagging
        # -------------------------
        if RUN_BAGGING:
            print("\nRunning Bagging...")

            best_params, best_val_acc, grid_df = tune_bagging(X_train, y_train, X_valid, y_valid)
            grid_path = os.path.join(RESULTS_DIR, f"bagging_grid_{dataset_id}.csv")
            safe_save_csv(grid_df, grid_path)

            test_acc, test_f1 = retrain_and_test_bagging(
                best_params, X_train, y_train, X_valid, y_valid, X_test, y_test
            )

            bag_summary_rows.append({
                "dataset": dataset_id,
                "n_estimators": best_params["n_estimators"],
                "max_depth": best_params["max_depth"],
                "best_validation_accuracy": best_val_acc,
                "test_accuracy": test_acc,
                "test_weighted_f1": test_f1
            })

            print("Best Bagging params:", best_params)
            print(f"Bagging validation accuracy: {best_val_acc:.4f}")
            print(f"Bagging test accuracy: {test_acc:.4f}")
            print(f"Bagging test weighted F1: {test_f1:.4f}")

            safe_save_csv(pd.DataFrame(bag_summary_rows), os.path.join(RESULTS_DIR, "bagging_summary.csv"))

        # -------------------------
        # Random Forest
        # -------------------------
        if RUN_RANDOM_FOREST:
            print("\nRunning Random Forest...")

            best_params, best_val_acc, grid_df = tune_random_forest(X_train, y_train, X_valid, y_valid)
            grid_path = os.path.join(RESULTS_DIR, f"random_forest_grid_{dataset_id}.csv")
            safe_save_csv(grid_df, grid_path)

            test_acc, test_f1 = retrain_and_test_random_forest(
                best_params, X_train, y_train, X_valid, y_valid, X_test, y_test
            )

            rf_summary_rows.append({
                "dataset": dataset_id,
                "n_estimators": best_params["n_estimators"],
                "max_features": best_params["max_features"],
                "best_validation_accuracy": best_val_acc,
                "test_accuracy": test_acc,
                "test_weighted_f1": test_f1
            })

            print("Best RF params:", best_params)
            print(f"RF validation accuracy: {best_val_acc:.4f}")
            print(f"RF test accuracy: {test_acc:.4f}")
            print(f"RF test weighted F1: {test_f1:.4f}")

            safe_save_csv(pd.DataFrame(rf_summary_rows), os.path.join(RESULTS_DIR, "random_forest_summary.csv"))

        # -------------------------
        # AdaBoost
        # -------------------------
        if RUN_ADABOOST:
            print("\nRunning AdaBoost...")

            best_params, best_val_acc, grid_df = tune_adaboost(X_train, y_train, X_valid, y_valid)
            grid_path = os.path.join(RESULTS_DIR, f"adaboost_grid_{dataset_id}.csv")
            safe_save_csv(grid_df, grid_path)

            test_acc, test_f1 = retrain_and_test_adaboost(
                best_params, X_train, y_train, X_valid, y_valid, X_test, y_test
            )

            ada_summary_rows.append({
                "dataset": dataset_id,
                "n_estimators": best_params["n_estimators"],
                "base_depth": best_params["base_depth"],
                "best_validation_accuracy": best_val_acc,
                "test_accuracy": test_acc,
                "test_weighted_f1": test_f1
            })

            print("Best AdaBoost params:", best_params)
            print(f"AdaBoost validation accuracy: {best_val_acc:.4f}")
            print(f"AdaBoost test accuracy: {test_acc:.4f}")
            print(f"AdaBoost test weighted F1: {test_f1:.4f}")

            safe_save_csv(pd.DataFrame(ada_summary_rows), os.path.join(RESULTS_DIR, "adaboost_summary.csv"))

    # =====================================================
    # FINAL COMPARISON TABLES
    # =====================================================

    if RUN_DECISION_TREE:
        dt_df = pd.DataFrame(dt_summary_rows)
    else:
        dt_df = pd.DataFrame()

    if RUN_BAGGING:
        bag_df = pd.DataFrame(bag_summary_rows)
    else:
        bag_df = pd.DataFrame()

    if RUN_RANDOM_FOREST:
        rf_df = pd.DataFrame(rf_summary_rows)
    else:
        rf_df = pd.DataFrame()

    if RUN_ADABOOST:
        ada_df = pd.DataFrame(ada_summary_rows)
    else:
        ada_df = pd.DataFrame()

    comparison_accuracy = pd.DataFrame({"dataset": dataset_ids})
    comparison_f1 = pd.DataFrame({"dataset": dataset_ids})

    if RUN_DECISION_TREE:
        comparison_accuracy = comparison_accuracy.merge(
            dt_df[["dataset", "test_accuracy"]].rename(columns={"test_accuracy": "DecisionTree"}),
            on="dataset",
            how="left"
        )
        comparison_f1 = comparison_f1.merge(
            dt_df[["dataset", "test_weighted_f1"]].rename(columns={"test_weighted_f1": "DecisionTree"}),
            on="dataset",
            how="left"
        )

    if RUN_BAGGING:
        comparison_accuracy = comparison_accuracy.merge(
            bag_df[["dataset", "test_accuracy"]].rename(columns={"test_accuracy": "Bagging"}),
            on="dataset",
            how="left"
        )
        comparison_f1 = comparison_f1.merge(
            bag_df[["dataset", "test_weighted_f1"]].rename(columns={"test_weighted_f1": "Bagging"}),
            on="dataset",
            how="left"
        )

    if RUN_RANDOM_FOREST:
        comparison_accuracy = comparison_accuracy.merge(
            rf_df[["dataset", "test_accuracy"]].rename(columns={"test_accuracy": "RandomForest"}),
            on="dataset",
            how="left"
        )
        comparison_f1 = comparison_f1.merge(
            rf_df[["dataset", "test_weighted_f1"]].rename(columns={"test_weighted_f1": "RandomForest"}),
            on="dataset",
            how="left"
        )

    if RUN_ADABOOST:
        comparison_accuracy = comparison_accuracy.merge(
            ada_df[["dataset", "test_accuracy"]].rename(columns={"test_accuracy": "AdaBoost"}),
            on="dataset",
            how="left"
        )
        comparison_f1 = comparison_f1.merge(
            ada_df[["dataset", "test_weighted_f1"]].rename(columns={"test_weighted_f1": "AdaBoost"}),
            on="dataset",
            how="left"
        )

    safe_save_csv(comparison_accuracy, os.path.join(RESULTS_DIR, "final_accuracy_table.csv"))
    safe_save_csv(comparison_f1, os.path.join(RESULTS_DIR, "final_f1_table.csv"))

    print_header("FINAL ACCURACY TABLE")
    print(comparison_accuracy)

    print_header("FINAL F1 TABLE")
    print(comparison_f1)

    elapsed = time.time() - start_time
    print_header("DONE")
    print(f"Total runtime: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()