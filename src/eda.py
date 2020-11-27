# author: Chen Zhao
# data: 2020-11-27

"""Creates eda plots for the pre-processed training data from 
the open hotel booking demand dataset 
(from https://www.sciencedirect.com/science/article/pii/S2352340918315191#f0010). 
Saves the plots as a pdf and png file.

Usage: src/eda.py --train=<train_data_file> --out_dir=<report_file>

Options:
--train=<train_data_file>     Path (including filename) to training data (which needs to be saved as an csv file)
--out_dir=<report_file>       Path to directory where the plots should be saved
"""

# common packages
import numpy as np
import pandas as pd

# Visualization packages
import altair as alt

# Save a vega-lite spec and a PNG blob for each plot in the notebook
alt.renderers.enable("mimetype")
# Handle large data sets without embedding them in the notebook
alt.data_transformers.enable("data_server")

opt = docopt(__doc__)


def main(train_data_file, report_file):

    train_df = pd.read_csv(train_data_file)
    X_train, y_train = train_df.drop(columns=["is_canceled"]), train_df["is_canceled"]

    # grab data:
    from pandas.api.types import CategoricalDtype

    prices_monthly = X_train[["hotel", "arrival_date_month", "adr_ac"]].sort_values(
        "arrival_date_month"
    )

    # order by month:
    months_ordered = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    prices_monthly["arrival_date_month"] = pd.Categorical(
        prices_monthly["arrival_date_month"], categories=months_ordered, ordered=True
    )
    prices_monthly = prices_monthly.sort_values("arrival_date_month")
    prices_points = (
        alt.Chart(prices_monthly, title="Room price per night over the year")
        .mark_point()
        .encode(
            alt.X("arrival_date_month", title="Month", sort=months_ordered),
            alt.Y("adr_ac", title="Price [EUR]"),
            alt.Color("hotel"),
        )
    )
    prices_points
    prices_points.mark_errorband(extent="ci") + prices_points.encode(
        y="mean(adr_ac)"
    ).mark_line()

    # Seperate Resort adn City Hotel:
    resort_train = X_train.loc[(X_train["hotel"] == "Resort Hotel")].copy()
    city_train = X_train.loc[(X_train["hotel"] == "City Hotel")].copy()

    rguests_monthly = resort_train.groupby("arrival_date_month")["hotel"].count()
    cguests_monthly = city_train.groupby("arrival_date_month")["hotel"].count()

    rguest_data = pd.DataFrame(
        {
            "month": list(rguests_monthly.index),
            "hotel": "Resort hotel",
            "guests": list(rguests_monthly.values),
        }
    )

    cguest_data = pd.DataFrame(
        {
            "month": list(cguests_monthly.index),
            "hotel": "City hotel",
            "guests": list(cguests_monthly.values),
        }
    )
    guest_data = pd.concat([rguest_data, cguest_data], ignore_index=True)

    # order by month:
    months_ordered = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    guest_data["month"] = pd.Categorical(
        guest_data["month"], categories=months_ordered, ordered=True
    )
    guest_data = guest_data.sort_values("month")
    # Dataset contains July and August date from 3 years, the other month from 2 years. Normalize data:
    guest_data.loc[
        (guest_data["month"] == "July") | (guest_data["month"] == "August"), "guests"
    ] /= 3
    guest_data.loc[
        ~((guest_data["month"] == "July") | (guest_data["month"] == "August")), "guests"
    ] /= 2
    guest_data

    guests_points = (
        alt.Chart(guest_data, title="Number of guests over the year")
        .mark_point()
        .encode(
            alt.X("month", title="Month", sort=months_ordered),
            alt.Y("guests", title="Number of guests"),
            alt.Color("hotel"),
        )
    )
    guests_points.mark_line()

    guests_prev_cancel = X_train[
        ["is_repeated_guest", "previous_bookings_not_canceled"]
    ]
    rep_guests_prev_cancel = (
        alt.Chart(guests_prev_cancel, title="Guests repeat booking with cancel history")
        .mark_bar()
        .encode(
            alt.X(
                "sum(previous_bookings_not_canceled)",
                title="Total number of previous bookings not cancelled",
            ),
            alt.Y("is_repeated_guest:O", title="Repeated guests"),
        )
    )
    rep_guests_prev_cancel


if __name__ == "__main__":
    main(opt["<train_data_file>"], opt["<report_file>"])
