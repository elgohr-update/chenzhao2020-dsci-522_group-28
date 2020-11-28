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
python src/classifier_evaluation.py data/processed/train_df.csv results/five_fold_cross_validation_result.csv --n_cv_folds=5 --chosen_seed=2020 --verbose=False

python src/model_tuning.py data/processed/train_df.csv results/ random_forest --n_iter=10 --n_cv_folds=5 --chosen_seed=2020 --verbose=False

## The prediction scripts below use a model file which is not on github due to size issue
## Please run the model_tuning.py script above to generate the model file
python src/predict_cancellation.py results/random_forest_model.sav data/processed/test_df.csv --result_path=results/ --model_name=random_forest --dataset_label=test --verbose=False

python src/predict_cancellation.py results/random_forest_model.sav data/processed/train_df.csv --result_path=results/ --model_name=random_forest --dataset_label=train --verbose=False

# render final report
Rscript -e "rmarkdown::render('doc/hotel_cancellation_predict_report.Rmd')"
