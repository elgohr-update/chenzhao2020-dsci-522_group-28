# author: Chen Zhao, Peter Yang
# data: 2020-11-27

"""Creates eda plots for the pre-processed training data from the open hotel booking demand dataset 
(from https://www.sciencedirect.com/science/article/pii/S2352340918315191#f0010). Saves the results 
as csv and svg files.

Usage: eda_ms2.py --train=<train_data_file> --out_dir=<report_file>

Options:
--train=<train_data_file>     Path (including filename) to training data (which needs to be saved as an csv file)
--out_dir=<report_file>       Path to directory where the plots should be saved
"""

# common packages
import numpy as np
import pandas as pd
from docopt import docopt

# Visualization packages
import altair as alt
import chromedriver_binary
from altair_saver import save
from selenium import webdriver

driver = webdriver.Chrome()


# Save a vega-lite spec and a PNG blob for each plot in the notebook
# alt.renderers.enable("mimetype")
# Handle large data sets without embedding them in the notebook
alt.data_transformers.enable("data_server")

opt = docopt(__doc__)


def main(train_data_file, report_file):

    # read in train data set
    try:
        train_df = pd.read_csv(train_data_file)
        print("Traning data set reading complete")
    except Exception as ex:
        print(ex)
        print(type(ex))
        exit(-99)

    # split features and target
    X_train, y_train = train_df.drop(columns=["is_canceled"]), train_df["is_canceled"]

    # Seperate Resort and City Hotel:
    resort_train = X_train.loc[(X_train["hotel"] == "Resort Hotel")].copy()
    city_train = X_train.loc[(X_train["hotel"] == "City Hotel")].copy()

    # numeric features distribution against target graph
    numeric_features = [
        "lead_time",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "booking_changes",
        "days_in_waiting_list",
        "adr",
        "required_car_parking_spaces",
        "total_of_special_requests",
    ]

    # categorical/numerical features against target graph
    categorical_features = [
        "hotel",
        "meal",
        "market_segment",
        "distribution_channel",
        "reserved_room_type",
        "deposit_type",
        "customer_type",
        "is_repeated_guest",
    ]

    train_df = train_df.copy()
    train_df["is_canceled_cat"] = train_df["is_canceled"].apply(
        lambda x: "Canceled" if x == 1 else "Not Canceled"
    )

    train_df['stays_in_weekend_nights'].value_counts()
    train_df_num = train_df.loc[train_df['stays_in_weekend_nights']<=10]
    train_df_num = train_df_num.loc[train_df_num['stays_in_week_nights']<=15]
    train_df_num = train_df_num.loc[train_df_num['lead_time']<=600]
    train_df_num = train_df_num.loc[train_df_num['adults']<=4]
    train_df_num = train_df_num.loc[train_df_num['children']<=3]
    train_df_num = train_df_num.loc[train_df_num['babies']<=2]
    train_df_num = train_df_num.loc[train_df_num['previous_cancellations']<=15]
    train_df_num = train_df_num.loc[train_df_num['previous_bookings_not_canceled']<=15]
    train_df_num = train_df_num.loc[train_df_num['booking_changes']<=10]
    train_df_num = train_df_num.loc[train_df_num['days_in_waiting_list']<=20]
    train_df_num = train_df_num.loc[train_df_num['adr']<=1000]

    numeric_vs_target = (alt.Chart(train_df_num)
    .mark_line(interpolate='monotone').encode(
        alt.X(alt.repeat(), type='quantitative'),
        alt.Y('count()', title = ""),
        alt.Color('is_canceled_cat', title = ""))).properties(width=150, height=150).repeat(numeric_features,columns = 4)

    try:
        numeric_vs_target.save(report_file + "/" + "numeric_vs_target.svg")
        print("Numeric features with target graph generating complete")
    except FileNotFoundError as fx:
        print("Error in target file path")
        print(fx)
        print(type(fx))
    except Exception as ex:
        print(ex)
        print(type(ex))
        
    def make_cat_graph(var):
    	test = train_df.groupby(var).agg({'is_canceled':'mean'})
    	test.reset_index(inplace = True)
    	test['is_canceled'] = round(test['is_canceled'], 2)
    	test[var] = test[var].astype(str)

    	graph = alt.Chart(test).mark_rect().encode(
        	alt.X(var),
        	alt.Color("is_canceled", title="Cancel Rate"),
    	).properties(width=300, height=200)
    
    	return graph+alt.Chart(test).mark_text(color = 'black').encode(alt.X(var), text = 'is_canceled')


    hotel = make_cat_graph('hotel')
    meal = make_cat_graph('meal')
    market_segment = make_cat_graph('market_segment')
    distribution_channel = make_cat_graph('distribution_channel')
    reserved_room_type = make_cat_graph('reserved_room_type')
    deposit_type = make_cat_graph('deposit_type')
    customer_type = make_cat_graph('customer_type')
    is_repeated_guest = make_cat_graph('is_repeated_guest')

    cat_vs_target = (hotel|meal|market_segment|distribution_channel)&(reserved_room_type|deposit_type|customer_type|is_repeated_guest)

    try:
        cat_vs_target.save(report_file + "/" + "cat_vs_target.svg")
        print("Categorical features with target graph generating complete")
    except FileNotFoundError as fx:
        print("Error in target file path")
        print(fx)
        print(type(fx))
    except Exception as ex:
        print(ex)
        print(type(ex))

    # missing table
    null_df = (
        train_df.isna()
        .sum()
        .reset_index(name="missing_count")
        .query("missing_count != 0")
    )
    null_df["missing_percentage"] = np.round(
        null_df["missing_count"] / train_df.shape[0] * 100, 2
    )
    null_df = null_df.rename({"index": "feature"}, axis=1)

    try:
        null_df.to_csv(report_file + "/" + "missing_summary.csv")
        print("Table with missing values generating complete")
    except FileNotFoundError as fx:
        print("Error in target file path")
        print(fx)
        print(type(fx))
    except Exception as ex:
        print(ex)
        print(type(ex))

    # correlation chart all variable
    corr_df = train_df.corr().stack().reset_index(name="corr")
    corr_df["round_corr"] = np.round(corr_df["corr"], 2)
    corr_plot = (
        alt.Chart(
            corr_df.query("level_0 != 'is_canceled' & level_1 != 'is_canceled'"),
            title="Feature Correlation",
        )
        .mark_rect()
        .encode(
            x="level_0",
            y="level_1",
            tooltip="corr",
            color=alt.Color(
                "corr", scale=alt.Scale(domain=(-1, 1), scheme="purpleorange")
            ),
        )
        .properties(width=500, height=500)
    )
    corr_text = (
        alt.Chart(corr_df.query("level_0 != 'is_canceled' & level_1 != 'is_canceled'"))
        .mark_text(size=8)
        .encode(
            x=alt.X("level_0", title="Features"),
            y=alt.Y("level_1", title="Features"),
            text="round_corr",
        )
        .properties(width=500, height=500)
    )
    corr_all = corr_plot + corr_text

    try:
        corr_all.save(report_file + "/" + "corr_all.svg")
        print("Feature correlation graph generating complete")
    except FileNotFoundError as fx:
        print("Error in target file path")
        print(fx)
        print(type(fx))
    except Exception as ex:
        print(ex)
        print(type(ex))

    # correlation against target chart
    corr_plot = (
        alt.Chart(
            corr_df[corr_df.level_1 == "is_canceled"], title="Feature Correlation"
        )
        .mark_rect()
        .encode(
            x=alt.X("level_0", title="Features"),
            y=alt.Y("level_1", title="Target"),
            tooltip="corr",
            color=alt.Color(
                "corr", scale=alt.Scale(domain=(-1, 1), scheme="purpleorange")
            ),
        )
        .properties(width=600)
    )

    corr_text = (
        alt.Chart(corr_df[corr_df.level_1 == "is_canceled"])
        .mark_text(size=8)
        .encode(
            x=alt.X("level_0", title="Features"),
            y=alt.Y("level_1", title="Target"),
            text="round_corr",
        )
        .properties(width=600)
    )
    corr_target = corr_plot + corr_text

    try:
        corr_target.save(report_file + "/" + "corr_target.svg")
        print("Feature correlation with target graph generating complete")
    except FileNotFoundError as fx:
        print("Error in target file path")
        print(fx)
        print(type(fx))
    except Exception as ex:
        print(ex)
        print(type(ex))

    # feature examination charts
    top_20_countries = (
        X_train.groupby("country")
        .size()
        .reset_index(name="counts")
        .sort_values(by="counts", ascending=False)[:20]
    )

    countries = (
        alt.Chart(top_20_countries, title="Top 20 home country of guests")
        .mark_bar()
        .encode(
            alt.X("counts", title="Guests numbers"),
            alt.Y("country", sort="-x", title="Country"),
            alt.Tooltip("country"),
        )
    )

    X_train["adr_ac"] = X_train["adr"] / (X_train["adults"] + X_train["children"])
    room_price = X_train[["hotel", "reserved_room_type", "adr_ac"]].sort_values(
        "reserved_room_type"
    )

    room_price = (
        alt.Chart(room_price)
        .mark_boxplot(extent="min-max", clip=True)
        .encode(
            alt.X("adr_ac", title="Price [EUR]", scale=alt.Scale(domain=(0, 120))),
            alt.Y("hotel", title="Hotel"),
            color="hotel",
        )
        .facet(
            "reserved_room_type",
            columns=2,
            title="Price per night and person for different room types",
        )
    )

    resort_train["total_nights"] = (
        resort_train["stays_in_weekend_nights"] + resort_train["stays_in_week_nights"]
    )
    city_train["total_nights"] = (
        city_train["stays_in_weekend_nights"] + city_train["stays_in_week_nights"]
    )

    num_nights_resort = list(resort_train["total_nights"].value_counts().index)
    num_bookings_resort = list(resort_train["total_nights"].value_counts())
    rel_bookings_resort = (
        resort_train["total_nights"].value_counts() / sum(num_bookings_resort) * 100
    )  # convert to percent

    num_nights_city = list(city_train["total_nights"].value_counts().index)
    num_bookings_city = list(city_train["total_nights"].value_counts())
    rel_bookings_city = (
        city_train["total_nights"].value_counts() / sum(num_bookings_city) * 100
    )  # convert to percent

    resort_nights = pd.DataFrame(
        {
            "hotel": "Resort hotel",
            "num_nights": num_nights_resort,
            "rel_num_bookings": rel_bookings_resort,
        }
    )

    city_nights = pd.DataFrame(
        {
            "hotel": "City hotel",
            "num_nights": num_nights_city,
            "rel_num_bookings": rel_bookings_city,
        }
    )

    nights_data = pd.concat([resort_nights, city_nights], ignore_index=True)
    nights_data

    stay = (
        alt.Chart(nights_data)
        .mark_bar()
        .encode(
            alt.X("num_nights", title="Number of nights"),
            alt.Y("rel_num_bookings", title="Percent of guests"),
            color=alt.Color("hotel", legend=None),
        )
        .facet("hotel", title="Length of guests stay")
    )

    feature_exam = (countries.properties(height=300, width=200) | stay) & room_price

    try:
        feature_exam.save(report_file + "/" + "feature_exam.svg")
        print("Feature examination chars generating complete")
    except FileNotFoundError as fx:
        print("Error in target file path")
        print(fx)
        print(type(fx))
    except Exception as ex:
        print(ex)
        print(type(ex))

    # price versus month graph
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
        .properties(width=500, height=400)
    )
    price_vs_month = prices_points.encode(y="mean(adr_ac)").mark_line()

    try:
        price_vs_month.save(report_file + "/" + "price_vs_month.svg")
        print("Room price changing graph generating complete")
    except FileNotFoundError as fx:
        print("Error in target file path")
        print(fx)
        print(type(fx))
    except Exception as ex:
        print(ex)
        print(type(ex))

    # guest versus month graph
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

    guests_points = (
        alt.Chart(guest_data, title="Number of guests over the year")
        .mark_point()
        .encode(
            alt.X("month", title="Month", sort=months_ordered),
            alt.Y("guests", title="Number of guests"),
            alt.Color("hotel"),
        )
        .properties(width=500, height=400)
    )
    guest_vs_month = guests_points.mark_line()

    try:
        guest_vs_month.save(report_file + "/" + "guest_vs_month.svg")
        print("Guest number changing graph generating complete")
    except FileNotFoundError as fx:
        print("Error in target file path")
        print(fx)
        print(type(fx))
    except Exception as ex:
        print(ex)
        print(type(ex))

    # guest repeat booking with cancel history graph
    guests_prev_cancel = X_train[
        ["is_repeated_guest", "previous_bookings_not_canceled"]
    ]
    rep_guests_prev_cancel = (
        alt.Chart(
            guests_prev_cancel, title="Guests repeat booking with cancellation history"
        )
        .mark_bar()
        .encode(
            alt.X(
                "sum(previous_bookings_not_canceled)",
                title="Total number of previous bookings not cancelled",
            ),
            alt.Y("is_repeated_guest:O", title="Repeated guests"),
        )
    )

    try:
        rep_guests_prev_cancel.save(report_file + "/" + "rep_guests_prev_cancel.svg")
        print("Repeat booking with cancellation history graph generating complete")
    except FileNotFoundError as fx:
        print("Error in target file path")
        print(fx)
        print(type(fx))
    except Exception as ex:
        print(ex)
        print(type(ex))

    print("Successfully generate EDA results")


if __name__ == "__main__":
    main(opt["--train"], opt["--out_dir"])
