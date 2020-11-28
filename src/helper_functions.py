# author: Debananda Sarkar
# date: 2020-11-26

"""This script hosts all the helper function for the project.
This will be imported in other scripts as required.
The main function of this script can be used to test the helper functions.

Usage: helper_function.py 

Options:
    None

"""

import pandas as pd
import numpy as np

def get_feature_lists():
    """This is a static function to DRY our code for feature categorization

    Returns
    -------
    A tuple of lists of feature types with feature names in them.
        numeric_features_general, 
        numeric_features_special, 
        categorical_features_general, 
        categorical_features_special, 
        drop_features, binary_features

    Examples
    --------
    >>> get_feature_lists()
    """
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
    numeric_features_special = ["children", "babies",]
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
    drop_features = ["company", "reservation_status", "reservation_status_date", "agent"]
    binary_features = ["is_repeated_guest"]
    return numeric_features_general, numeric_features_special, categorical_features_general, categorical_features_special, drop_features, binary_features

def summarize_cv_scores(X, classifier_name):

    """Formats the output of cross_validate function from sklearn.model_selection
    from dictionary to pandas Dataframe

    Parameters
    ----------
    X : dict
        A dictionary which holds the output of sklearn.model_selection.cross_validate function

    classifier_name: str
        Name of the classifier used for sklearn.model_selection.cross_validate function

    Returns
    -------
    X_df : pandas.DataFrame
        A dataframe with model name and cross validation statistics

    Examples
    --------
    >>> toy_score = {'fit_time': np.array([0.02657628, 0.02356863, 0.02640152, 0.02609539, 0.02393484]),
        'score_time': np.array([0.02375078, 0.02024961, 0.02282763, 0.02582216, 0.0217247 ]),
        'test_accuracy': np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        'train_accuracy': np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        'test_f1': np.array([0., 0., 0., 0., 0.]),
        'train_f1': np.array([0., 0., 0., 0., 0.]),
        'test_recall': np.array([0., 0., 0., 0., 0.]),
        'train_recall': np.array([0., 0., 0., 0., 0.]),
        'test_precision': np.array([0., 0., 0., 0., 0.]),
        'train_precision': np.array([0., 0., 0., 0., 0.]),
        'test_average_precision': np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        'train_average_precision': np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        'test_roc_auc': np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        'train_roc_auc': np.array([0.5, 0.5, 0.5, 0.5, 0.5])}

    >>> summarize_cv_scores(toy_score, "toy_test")
    """

    X_df = pd.DataFrame(X)
    col_names = (
        pd.Series(X_df.columns.tolist()).str.replace("test_", "validation_").tolist()
    )
    X_df = pd.DataFrame(X_df.mean()).T
    X_df.columns = col_names
    X_df["classifier_name"] = classifier_name
    col_names = ["classifier_name"] + col_names
    return X_df[col_names]


def main():

    ###################### Test for summarize_cv_scores ###########################

    toy_score = {
        "fit_time": np.array(
            [0.02657628, 0.02356863, 0.02640152, 0.02609539, 0.02393484]
        ),
        "score_time": np.array(
            [0.02375078, 0.02024961, 0.02282763, 0.02582216, 0.0217247]
        ),
        "test_accuracy": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        "train_accuracy": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        "test_f1": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "train_f1": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "test_recall": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "train_recall": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "test_precision": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "train_precision": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "test_average_precision": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        "train_average_precision": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        "test_roc_auc": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        "train_roc_auc": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
    }

    expected = {
        "classifier_name": ["toy_test"],
        "validation_accuracy": [0.5],
        "train_accuracy": [0.5],
        "validation_f1": [0.0],
        "train_f1": [0.0],
        "validation_recall": [0.0],
        "train_recall": [0.0],
        "validation_precision": [0.0],
        "train_precision": [0.0],
        "validation_average_precision": [0.5],
        "train_average_precision": [0.5],
        "validation_roc_auc": [0.5],
        "train_roc_auc": [0.5],
    }


    assert isinstance(summarize_cv_scores(toy_score, "toy_test"), pd.DataFrame), "Check data structure"
    assert int((summarize_cv_scores(toy_score, "toy_test").loc[
        :,
        [
            "classifier_name",
            "validation_accuracy",
            "train_accuracy",
            "validation_f1",
            "train_f1",
            "validation_recall",
            "train_recall",
            "validation_precision",
            "train_precision",
            "validation_average_precision",
            "train_average_precision",
            "validation_roc_auc",
            "train_roc_auc",
        ],
    ] == pd.DataFrame(data=expected)).T.sum())==13, "Check function logic"
    print("Successfully tested summarize_cv_scores function!")


if __name__ == "__main__":
    main()
