# DSCI-522_group-28

  - contributors: Jared Splinter, Chen Zhao, Debananda Sarkar, Peter Yang

Data analysis project for DSCI 522 (Data Science workflows); a
course in the Master of Data Science program at the University of
British Columbia.

## Introduction

For this project we are trying to answer the question: given a hotel booking, is the reservation likely to be cancelled? Cancelled bookings can have a big impact on the hospitality industry and hotels would like to get an estimate if a booking is likely to be cancelled. When a hotel's booking is cancelled, the hotel loses out on the revenue that booking would have brought as well as on potential other bookings that could have replaced the cancelled booking. Thus, predicting cancellations is useful for a hotel's revenue management. Finding the conditions on which a booking is likely to be cancelled can help a hotel improve the conditions and limit the number of cancellations they receive, thereby increasing their revenue.

The data set used in this project comes from the Hotel Booking demand datasets from [Antonio, Almeida and Nunes, 2019](https://www.sciencedirect.com/science/article/pii/S2352340918315191#ack0005) and the data can be found from the GitHub Repository [here](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11). The dataset contains real world data obtained from two hotels; one resort hotel and one city hotel. Each row represents an individual hotel booking due to arrive between July 1st, 2015 and August 31st, 2017. There are 31 columns describing 40,060 observations from the resort hotel and 79,330 observations from the city hotel totalling 119,390 bookings.

To answer the predictive question, we plan to build a predictive supervised learning classification model. We will start by splitting the data into an 80%:20% training and test set, this will give lots of train data for us to perform cross-validation. Next, we will perform exploratory data analysis to first determine whether there is a strong class imbalance problem, as well as explore whether there are any features that would not be useful for prediction that we should omit. Such features could include distributions that appear similar between the two classes, and features that add no potential use.

In our preliminary EDA, we will check if we have sufficient observations for both hotel types (resort and city). Then we will check if we have class imbalance issue and try to decide the most suitable evaluation metrics. Before going into in-depth analysis of the features, we will check first few records from the dataset, their datatypes any the summary of missing values. This will be followed by a quick look at the correlation between the features. We will also look into the correlation between features and the target to get a overview about the features which may have impact of the target prediction.

Based on our finding from the above we will scope our next steps in data analysis.

- We will try to identify if those explanatory features whose distributions look different for the target classes and perform thorough analysis of those features.
- We will try to identify or engineer new feature which may influence the prediction.
- We will also try to identify how each features are related to each other.

Given that our prediction target is one of two classes, we will choose a classification model. We will test many models to find one that gives good accuracy but one we suspect will be a good model is Logistic Regression as this can give us access to feature weights which would be useful in analysis. This is assuming the question can be answered when decision boundaries are linear. Once we have picked our model we will run hyperparameter optimization using RandomizedSearchCV to get the best scores. If there is a class imbalance we will include scores for precision, recall and f1. As the dataset is very large we do not expect to run into issues such as optimization bias. 

After selecting our final model, we will re-fit the model on the entire training data set. We will then score the model on a series of tests depending on if class imbalance is found and also look at overall accuracy as well as misclassification errors. These values will be reported as a table in the final report.

## Usage

To replicate the analysis, clone this GitHub repository, install the
[dependencies](#dependencies) listed below, and run the following
commands at the command line/terminal from the root directory of this
project:

   python src/getdata.py --source_url=https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv --target_file=data/raw/hotels_dataset.csv
   python src/split_dataset_train_test.py --source_data_file=data/raw/hotels_dataset.csv --train_data_file=data/processed/train_df.csv --test_data_file=data/processed/test_df.csv --test_split_ratio=0.2 --chosen_seed=2020

## Dependencies

(Update Dependencies )

  - Python 3.8.6 and Python packages:
      - docopt==0.6.2
      - pandas==1.1.4
      - sklearn==0.23.2
      - altair==4.1.0
      - altair_saver==0.1.0
      - numpy==1.19.4

## License

The Hotel Booking demand dataset is an open access article distributed under the terms of the Creative Commons CC-BY license, which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited.  If re-using/re-mixing please provide attribution and link to this webpage.

# References

<div id="refs" class="references">

<div id="ref-Hotel2019">

Antonio, Nuno, Ana de Almeida, and Luis Nunes. 2019. "Hotel booking demand datasets." Data in brief 22: 41-49. <https://doi.org/10.1016/j.dib.2018.11.126>

</div>

</div>
