# DSCI-522_group-28

  - contributors: Jared Splinter, Chen Zhao, Debananda Sarkar, Peter Yang

Data analysis project for DSCI 522 (Data Science workflows); a
course in the Master of Data Science program at the University of
British Columbia.

## About

Here we attempt to build a classification model to predict whether a given hotel booking is likely to be canceled. A model was selected by comparing many classification algorithims and selecting the best one as the Random Forest classification algorithm. From there, hyperparameter optimization was performed and the best resulting model was selected. Our final model was scored using f1 metric on the test data and received a 0.835 compared to the train data f1 score of 0.983. The model incorrectly predicted 11.4% of the test bookings and suggests that the final model has been overfit. The model as it is may help hotels with their revenue management however we recommend continued improvement of the model and further feature examination/engineering to reduce overfitting and improve test score.

The data set used in this project comes from the Hotel Booking demand datasets from Antonio, Almeida and Nunes at Instituto Universitário de Lisboa (ISCTE-IUL), Lisbon, Portugal  (Antonio, Almeida, and Nunes 2019). The data was sourced directly from the Github Repository [here](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11). The dataset contains real world data obtained from two hotels; one resort hotel and one city hotel. Each row represents an individual hotel booking due to arrive between July 1st, 2015 and August 31st, 2017. There are 31 columns describing 40,060 observations from the resort hotel and 79,330 observations from the city hotel totaling 119,390 bookings.


## Reports

The EDA report can be found
[here](https://github.com/UBC-MDS/dsci-522_group-28/blob/main/src/hotels_data_preliminary_analysis_eda.ipynb).

The final report can be found
[here](https://github.com/ttimbers/breast_cancer_predictor/blob/v2.0/doc/breast_cancer_predict_report.md).

## Usage

To replicate the analysis, clone this GitHub repository, install the [dependencies](#dependencies) listed below, and run the [runall.sh](https://github.com/UBC-MDS/dsci-522_group-28/blob/main/runall.sh) file or run the following commands at the command line/terminal from the root directory of this project:


```bash
# download data set
python src/getdata.py --source_url=https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv --target_file=data/raw/hotels_dataset.csv

# preprocess data
python src/split_dataset_train_test.py --source_data_file=data/raw/hotels_dataset.csv --train_data_file=data/processed/train_df.csv --test_data_file=data/processed/test_df.csv --test_split_ratio=0.2 --chosen_seed=2020

# create exploratory data analysis tables and figures, and write to file
python src/eda_ms2.py --train=data/processed/train_df.csv --out_dir=results/

# train model and tune hyperparameters
python src/classifier_evaluation.py data/processed/train_df.csv results/five_fold_cross_validation_result.csv --n_cv_folds=5 --chosen_seed=2020 --verbose=False

python src/model_tuning.py data/processed/train_df.csv results/ random_forest --n_iter=10 --n_cv_folds=5 --chosen_seed=2020 --verbose=False

## The prediction scripts below use a model file which is not on github due to size issue
## Please run the model_tuning.py script above to generate the model file
python src/predict_cancellation.py results/random_forest_model.sav data/processed/test_df.csv --result_path=results/ --model_name=random_forest --dataset_label=test --verbose=False

python src/predict_cancellation.py results/random_forest_model.sav data/processed/train_df.csv --result_path=results/ --model_name=random_forest --dataset_label=train --verbose=False

# render final report
Rscript -e "rmarkdown::render('doc/hotel_cancellation_predict_report.Rmd')"

```

## Dependencies


  - Python 3.8.6 and Python packages:
      - docopt==0.6.2
      - pandas==1.1.4
      - sklearn==0.23.2
      - altair==4.1.0
      - altair_saver==0.1.0
      - numpy==1.19.4
      - pickle==4.0
      - matplotlib==3.3.3
      - selenium = 3.141.0
      - python-chromedriver-binary = 87.0.4280.20.0
  - R==4.0.3
      - Knitr==1.29
      - KableExtra==1.3.1
      - tidyverse==1.3.0

## License

The Hotel Booking demand dataset is an open access article distributed under the terms of the Creative Commons CC-BY license, which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited.  If re-using/re-mixing please provide attribution and link to this webpage.

# References

<div id="refs" class="references">

<div id="ref-Hotel2019">

Antonio, Nuno, Ana de Almeida, and Luis Nunes. 2019. "Hotel booking demand datasets." Data in brief 22: 41-49. <https://doi.org/10.1016/j.dib.2018.11.126>

</div>

</div>
