# author: Debananda Sarkar
# date: 2020-11-27

"""This script will tune hyperparameter for of the model
using random search algorithm from scikit-learn

Usage: model_tuning.py <train_data_file> <report_path> <model_name> [--n_iter=<n_iter>] [--n_cv_folds=<n_cv_folds>] [--chosen_seed=<chosen_seed>] [--verbose=<verbose>]

Options:
<train_data_file>               Relative path to training data file
<report_path>                   Relative path to report folder where the result and model will be saved (must end with "/")
<model_name>                    Model name to be used. Should be one of the following: [decision_tree, knn, svc, logistic_regression, random_forest]
[--n_iter=<n_iter>]             Number of search iterations to be used by random search algorithm [Optional, default = 10]
[--n_cv_folds=<n_cv_folds>]     Number of cross validation folds to be used [Optional, default = 5]
[--chosen_seed=<chosen_seed>]   Seed value to be used [Optional, default = 1]
[--verbose=<verbose>]           Prints out message if True [Optional, default = False]
"""

from docopt import docopt
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from helper_functions import summarize_cv_scores, get_feature_lists, get_hyperparameter_grid
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

opt = docopt(__doc__)


def main(
    train_data_file,
    report_path,
    model_selected,
    n_search_iterations,
    n_cv_folds,
    chosen_seed,
    verbose,
):

    assert model_selected in [
        "decision_tree",
        "knn",
        "svc",
        "logistic_regression",
        "random_forest",
    ], "Invalid model name..."

    if verbose == "True":
        verbose = True
    else:
        verbose = False

    try:
        if chosen_seed is None:
            chosen_seed = 1
        else:
            chosen_seed = int(chosen_seed)
    except ValueError as vx:
        print("Value of chosen_seed should be int")
        print(vx)
        print(type(vx))
        exit(-2)
    except Exception as ex:
        print(ex)
        print(type(ex))
        exit(-99)

    try:
        if n_cv_folds is None:
            n_cv_folds = 5
        else:
            n_cv_folds = int(n_cv_folds)
    except ValueError as vx:
        print("Value of n_cv_folds should be int")
        print(vx)
        print(type(vx))
        exit(-3)
    except Exception as ex:
        print(ex)
        print(type(ex))
        exit(-99)

    try:
        if n_search_iterations is None:
            n_search_iterations = 10
        else:
            n_search_iterations = int(n_search_iterations)
    except ValueError as vx:
        print("Value of n_search_iterations should be int")
        print(vx)
        print(type(vx))
        exit(-4)
    except Exception as ex:
        print(ex)
        print(type(ex))
        exit(-99)

    train_df = pd.read_csv(train_data_file)
    if verbose:
        print("Training Data Imported...")
    X_train, y_train = train_df.drop(columns=["is_canceled"]), train_df["is_canceled"]

    (
        numeric_features_general,
        numeric_features_special,
        categorical_features_general,
        categorical_features_special,
        drop_features,
        binary_features,
    ) = get_feature_lists()

    numeric_pipeline_general = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )
    numeric_pipeline_special = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0), StandardScaler()
    )
    categorical_pipeline_general = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
    )
    categorical_pipeline_special = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    binary_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"))

    preprocessor = make_column_transformer(
        (numeric_pipeline_general, numeric_features_general),
        (numeric_pipeline_special, numeric_features_special),
        (categorical_pipeline_general, categorical_features_general),
        (categorical_pipeline_special, categorical_features_special),
        (binary_pipeline, binary_features),
    )

    models = {
        "decision_tree": DecisionTreeClassifier(random_state=chosen_seed),
        "knn": KNeighborsClassifier(n_jobs=-1, n_neighbors=3),
        "svc": SVC(random_state=chosen_seed, verbose=True),
        "logistic_regression": LogisticRegression(
            random_state=chosen_seed, max_iter=1000, n_jobs=-1
        ),
        "random_forest": RandomForestClassifier(n_jobs=-1, random_state=chosen_seed),
    }

    model_pipe = make_pipeline(preprocessor, models[model_selected])

    param_grid = get_hyperparameter_grid(model_selected)

    if verbose:
        verbose_level=10
    else:
        verbose_level=0

    random_search = RandomizedSearchCV(
        model_pipe,
        param_distributions=param_grid,
        n_iter=n_search_iterations,
        scoring="f1",
        n_jobs=-1,
        cv=n_cv_folds,
        verbose=verbose_level,
        random_state=chosen_seed,
        return_train_score=True,
    )

    random_search.fit(X_train, y_train)

    print(f"Best parameter: {random_search.best_params_}")
    print(f"Best validation score: {round(random_search.best_score_, 3)}")

    cv_scores = pd.DataFrame(random_search.cv_results_)
    cv_scores.to_csv(f"{report_path}{model_selected}_tuning_result.csv", index=False)

    final_model = random_search.best_estimator_
    final_model.fit(X_train, y_train)

    pickle.dump(
        final_model,
        open(f"{report_path}{model_selected}_model.sav", "wb"),
    )

    print("Tuning complete...")

if __name__ == "__main__":
    main(
        opt["<train_data_file>"],
        opt["<report_path>"],
        opt["<model_name>"],
        opt["--n_iter"],
        opt["--n_cv_folds"],
        opt["--chosen_seed"],
        opt["--verbose"],
    )
