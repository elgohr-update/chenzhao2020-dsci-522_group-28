# author: Debananda Sarkar
# date: 2020-11-27

"""This script will evaluate the performance of 
various classification algorithm on training data using 
cross validation

Usage: classifier_evaluation.py <train_data_file> <report_file> [--n_cv_folds=<n_cv_folds>] [--chosen_seed=<chosen_seed>] [--verbose=<verbose>]

Options:
<train_data_file>               Relative path to training data file
<report_file>                   Relative path to cross validation report file
[--n_cv_folds=<n_cv_folds>]     Number of cross validation folds to be used [Optional, default = 2]
[--chosen_seed=<chosen_seed>]   Seed value to be used [Optional, default = 1]
[--verbose=<verbose>]           Prints out message if True [Optional, default = False]
"""

from docopt import docopt
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from helper_functions import summarize_cv_scores

opt = docopt(__doc__)


def main(train_data_file, report_file, n_cv_folds, chosen_seed, verbose):

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
            n_cv_folds = 2
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

    train_df = pd.read_csv(train_data_file)
    if verbose:
        print("Training Data Imported...")
    X_train, y_train = train_df.drop(columns=["is_canceled"]), train_df["is_canceled"]

    numeric_features_general = [
        "lead_time",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "booking_changes",
        "days_in_waiting_list",
        "adr",
        "required_car_parking_spaces",
        "total_of_special_requests",
        "arrival_date_year",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
    ]
    numeric_features_special = [
        "children",
        "babies",
    ]
    categorical_features_general = [
        "hotel",
        "arrival_date_month",
        "meal",
        "market_segment",
        "distribution_channel",
        "reserved_room_type",
        "deposit_type",
        "customer_type",
    ]
    categorical_features_special = ["country"]
    drop_features = [
        "company",
        "reservation_status",
        "reservation_status_date",
        "agent",
    ]
    binary_features = ["is_repeated_guest"]

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
    if verbose:
        print("Preprocessor Created...")
    eval_metrics = ["f1", "precision", "recall", "accuracy"]

    model_dummy = DummyClassifier(strategy="stratified", random_state=chosen_seed)
    scores = cross_validate(
        model_dummy,
        X_train,
        y_train,
        scoring=eval_metrics,
        cv=n_cv_folds,
        n_jobs=-1,
        return_train_score=True,
    )

    mean_scores_df = summarize_cv_scores(scores, "Dummy Classifier")

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=chosen_seed),
        # "Naive-Bayes": MultinomialNB(),
        "k_Nearest_Neighbor": KNeighborsClassifier(n_jobs=-1, n_neighbors = 3),
        "SVC (RBF kernel)": SVC(random_state=chosen_seed),
        "Logistic Regression": LogisticRegression(
            random_state=chosen_seed, max_iter=1000, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(n_jobs=-1, random_state=chosen_seed)
    }

    for m in models:
        if verbose:
            print(f"Evaluating {m}")
        model_pipe = make_pipeline(preprocessor, models[m])
        scores = cross_validate(
            model_pipe,
            X_train,
            y_train,
            scoring=eval_metrics,
            cv=n_cv_folds,
            n_jobs=-1,
            return_train_score=True,
        )
        mean_scores_df = pd.concat(
            [mean_scores_df, summarize_cv_scores(scores, m)],
            ignore_index=True,
        )
    if verbose:
        print("Evaluation Complete...")

    mean_scores_df.to_csv(report_file, index=False)

    print("Report generated and saved...")


if __name__ == "__main__":
    main(
        opt["<train_data_file>"],
        opt["<report_file>"],
        opt["--n_cv_folds"],
        opt["--chosen_seed"],
        opt["--verbose"],
    )
