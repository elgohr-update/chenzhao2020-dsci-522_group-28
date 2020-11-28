# Chen Zhao
# 2020-11-27
# 
# This driver script is for predicting if a reservation is likely to be cancelled by given a hotel booking detail.
#
# Usage:
# bash runall.sh

# download data set
python src/getdata.py --source_url=https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv --target_file=data/raw/hotels_dataset.csv

# preprocess data
python src/split_dataset_train_test.py --source_data_file=data/raw/hotels_dataset.csv --train_data_file=data/processed/train_df.csv --test_data_file=data/processed/test_df.csv --test_split_ratio=0.2 --chosen_seed=2020

# create exploratory data analysis tables and figures, and write to file
python src/eda_ms2.py --train=data/processed/train_df.csv --out_dir=results/

# train model and tune hyperparameters
python src/classifier_evaluation.py data/processed/train_df.csv reports/two_fold_cross_validation_result.csv --n_cv_folds=2 --chosen_seed=2020 --verbose=True
python src/classifier_evaluation.py data/processed/train_df.csv reports/five_fold_cross_validation_result.csv --n_cv_folds=5 --chosen_seed=2020 --verbose=True

# render final report
