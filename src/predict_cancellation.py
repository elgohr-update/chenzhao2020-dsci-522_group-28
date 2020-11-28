# author: Debananda Sarkar
# date: 2020-11-28

"""This script will tune hyperparameter for of the model
using random search algorithm from scikit-learn

Usage: predict_cancellation.py <model_path> <dataset_path> --result_path=<result_path> --model_name=<model_name> --dataset_label=<dataset_label> [--verbose=<verbose>]

Options:
<model_path>                        Relative path of the model file
<dataset_path>                      Relative path of the data file to be used for prediction
--result_path=<result_path>         Relative path to the folder where the results will be saved (must end with "/")
--model_name=<model_name>           A string label for the model. This will be used in report file names and labelling
--dataset_label=<dataset_label>     Dataset description. Should be one of the following: [train, test, validation, deployment]
[--verbose=<verbose>]               Prints out message if True [Optional, default = False]

"""

from docopt import docopt
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from helper_functions import summarize_cv_scores, get_feature_lists
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    plot_confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    plot_precision_recall_curve
)
import matplotlib.pyplot as plt
# import altair as alt

opt = docopt(__doc__)


def main(
    model_path,
    dataset_path,
    result_path,
    model_selected,
    dataset_label,
    verbose
):

    assert dataset_label in ["train", "test", "validation", "deployment"], "Invalid dataset_label used."

    if verbose == "True":
        verbose = True
    else:
        verbose = False

    loaded_model = pickle.load(open(model_path, 'rb'))
    if verbose:
        print("Model Loaded...")

    data_df = pd.read_csv(dataset_path)
    if verbose:
        print(f"{dataset_label} data loaded...")

    X, y = data_df.drop(columns=["is_canceled"]), data_df["is_canceled"]
    y_pred = loaded_model.predict(X)
    print(f"Prediction completed...")

    cm_plot = plot_confusion_matrix(
        loaded_model,
        X,
        y,
        values_format="d",
        cmap=plt.cm.Blues,
        display_labels=["0 - Not Canceled", "1 - Canceled"],
    );
    plt.savefig(f"{result_path}{model_selected}_confusion_matrix_{dataset_label}_data.png")

    prc = plot_precision_recall_curve(loaded_model, X, y, name=f"{dataset_label} data")
    plt.plot(
        recall_score(y, y_pred),
        precision_score(y, y_pred),
        "or",
        markersize=10,
    );
    plt.savefig(f"{result_path}{model_selected}_precision_recall_curve_{dataset_label}_data.png")

    score_summary = pd.DataFrame(
        data={
            "dataset": f"{dataset_label}_data",
            "f1_score": f1_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "accuracy": accuracy_score(y, y_pred),
        },
        index=[0],
    )
    score_summary.to_csv(f"{result_path}{model_selected}_score_summary_{dataset_label}_data.csv", index=False)

    '''
    cm_table = pd.DataFrame(
        confusion_matrix(y, y_pred),
        columns=["0 - Not Canceled", "1 - Canceled"],
        index=["0 - Not Canceled", "1 - Canceled"],
    ).stack().reset_index(name="value").rename(
        columns={"level_0": "true_label", "level_1": "predicted_label"}
    )

    cm_plot = (
        alt.Chart(cm_table, title="Confusion Matrix")
        .mark_rect()
        .encode(
            x=alt.X(
                "predicted_label:N", axis=alt.Axis(labelAngle=-45), title="Predicted Label"
            ),
            y=alt.Y("true_label:N", title="True Label"),
            color=alt.Color("value", scale=alt.Scale(scheme="blues"), title="")
        )
        .properties(width=250, height=250)
    )

    cm_text = (
        alt.Chart(cm_table)
        .mark_text(size=15, fontStyle="bold")
        .encode(
            x="predicted_label:N",
            y="true_label:N",
            text="value",
        )
        .properties(width=250, height=250)
    )
    
    cm_final_plot = cm_plot + cm_text
    cm_final_plot.save(f"{result_path}{model_selected}_confusion_matrix_{dataset_label}_data.svg")
    '''


    print("Results saved...")

if __name__ == "__main__":
    main(
        opt["<model_path>"],
        opt["<dataset_path>"],
        opt["--result_path"],
        opt["--model_name"],
        opt["--dataset_label"],
        opt["--verbose"],
    )
