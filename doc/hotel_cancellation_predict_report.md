Hotel Cancellation Predict Report
================
Jared Splinter
11/27/2020

  - [Predicting Hotel Booking Cancellation from Real World Hotel
    Bookings](#predicting-hotel-booking-cancellation-from-real-world-hotel-bookings)
  - [Summary](#summary)
  - [Introduction](#introduction)
  - [Methods](#methods)
      - [Data](#data)
      - [Analysis](#analysis)
  - [Results & Discussion](#results-discussion)
  - [References](#references)

# Predicting Hotel Booking Cancellation from Real World Hotel Bookings

Jared Splinter,

Created: 11/27/2020

# Summary

# Introduction

The hospitality industry and hotels in particular suffer huge revenue
losses due to booking cancellations and no shows. The revenue lost
becomes a sunk cost when there is not enough time to book the room again
before the date of stay (Xie and Gerstner 2007). Hotels would like to
get an estimate if a booking is likely to be cancelled as predicting
cancellations is useful for a hotel’s revenue management.

Here we ask if we can use a machine learning algorithm to predict
whether a given hotel booking is likely to be cancelled. Finding the
conditions on which a booking is likely to be cancelled can help a hotel
improve the conditions and limit the number of cancellations they
receive, thereby increasing their revenue. If a booking is likely to be
cancelled a hotel may also wish to implement higher cancellation fees to
make up some of the lost revenue (Chen, Schwartz, and Vargas 2011). If a
machine learning algothrithm can accurately predict if a hotel booking
will be cancelled it could help hotels make up some of their lost
revenue and potentially find ways in which to improve customer
satisfaction.

# Methods

## Data

The data set used in this project comes from the Hotel Booking demand
datasets from Antonio, Almeida and Nunes at Instituto Universitário de
Lisboa (ISCTE-IUL), Lisbon, Portugal (Antonio, Almeida, and Nunes 2019).
The data was sourced directly from the Github Repository
[here](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11).
The dataset contains real world data obtained from two hotels; one
resort hotel and one city hotel. Each row represents an individual hotel
booking due to arrive between July 1st, 2015 and August 31st, 2017.
There are 31 columns describing 40,060 observations from the resort
hotel and 79,330 observations from the city hotel totalling 119,390
bookings.

## Analysis

Many classification model algorithms were compared using
cross-validation so the best classification model could be selected. 5
fold cross validation was selected as the data set is quite large and it
was scored on f1, precision, recall and accuracy as there is class
imbalance within the dataset. f1 scores are reported as it is a good
balance between recall and precision scores. The classification models
compared are: Dummy Classifier, Decision Tree, K-nearest neighbor, SVC
with RBF kernel, Logistic Regression, and Random Forest.

From there, Random Forest was chosen as the classification model and
hyperparameter optimization was carried out using Random Search
Cross-Validation. The hyperparameters optimized from Random Forest were
`n_estimators` and `min_sample_split`. The best model from the Random
Search Cross-validation was selected and used to fit on the train data
and then to score on the test data. 4 features were dropped from the
analysis: company, agent, reservation\_status, and
reservation\_status\_date.

The Python programming language (Van Rossum and Drake 2009) and the
following Python packages were used to perform the analysis: docopt
(Keleshev 2014), pandas (McKinney 2010; team 2020), sklearn (Pedregosa
et al. 2011), altair (VanderPlas et al. 2018), numpy (Harris et al.
2020). The code used to perform the analysis and create this report can
be found here: <https://github.com/UBC-MDS/dsci-522_group-28>

# Results & Discussion

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">

<caption>

Table 1. A table that I have inserted

</caption>

<thead>

<tr>

<th style="text-align:left;">

classifier\_name

</th>

<th style="text-align:right;">

fit\_time

</th>

<th style="text-align:right;">

score\_time

</th>

<th style="text-align:right;">

validation\_f1

</th>

<th style="text-align:right;">

train\_f1

</th>

<th style="text-align:right;">

validation\_precision

</th>

<th style="text-align:right;">

train\_precision

</th>

<th style="text-align:right;">

validation\_recall

</th>

<th style="text-align:right;">

train\_recall

</th>

<th style="text-align:right;">

validation\_accuracy

</th>

<th style="text-align:right;">

train\_accuracy

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

Dummy Classifier

</td>

<td style="text-align:right;">

0.0193955

</td>

<td style="text-align:right;">

0.0223241

</td>

<td style="text-align:right;">

0.3706303

</td>

<td style="text-align:right;">

0.3720946

</td>

<td style="text-align:right;">

0.3717293

</td>

<td style="text-align:right;">

0.3714800

</td>

<td style="text-align:right;">

0.3695377

</td>

<td style="text-align:right;">

0.3727114

</td>

<td style="text-align:right;">

0.5342261

</td>

<td style="text-align:right;">

0.5331634

</td>

</tr>

<tr>

<td style="text-align:left;">

Decision Tree

</td>

<td style="text-align:right;">

19.0155343

</td>

<td style="text-align:right;">

0.0980628

</td>

<td style="text-align:right;">

0.7940997

</td>

<td style="text-align:right;">

0.9945951

</td>

<td style="text-align:right;">

0.7912478

</td>

<td style="text-align:right;">

0.9963554

</td>

<td style="text-align:right;">

0.7969927

</td>

<td style="text-align:right;">

0.9928414

</td>

<td style="text-align:right;">

0.8466057

</td>

<td style="text-align:right;">

0.9959953

</td>

</tr>

<tr>

<td style="text-align:left;">

k\_Nearest\_Neighbor

</td>

<td style="text-align:right;">

17.4371542

</td>

<td style="text-align:right;">

194.6145562

</td>

<td style="text-align:right;">

0.7645612

</td>

<td style="text-align:right;">

0.8772294

</td>

<td style="text-align:right;">

0.7719782

</td>

<td style="text-align:right;">

0.8904962

</td>

<td style="text-align:right;">

0.7573279

</td>

<td style="text-align:right;">

0.8643538

</td>

<td style="text-align:right;">

0.8268909

</td>

<td style="text-align:right;">

0.9102102

</td>

</tr>

<tr>

<td style="text-align:left;">

SVC (RBF kernel)

</td>

<td style="text-align:right;">

506.2891913

</td>

<td style="text-align:right;">

61.0231221

</td>

<td style="text-align:right;">

0.7905805

</td>

<td style="text-align:right;">

0.8047185

</td>

<td style="text-align:right;">

0.8567910

</td>

<td style="text-align:right;">

0.8699969

</td>

<td style="text-align:right;">

0.7338842

</td>

<td style="text-align:right;">

0.7485542

</td>

<td style="text-align:right;">

0.8557040

</td>

<td style="text-align:right;">

0.8651688

</td>

</tr>

<tr>

<td style="text-align:left;">

Logistic Regression

</td>

<td style="text-align:right;">

14.0690587

</td>

<td style="text-align:right;">

0.0916533

</td>

<td style="text-align:right;">

0.7207729

</td>

<td style="text-align:right;">

0.7226086

</td>

<td style="text-align:right;">

0.8088454

</td>

<td style="text-align:right;">

0.8106400

</td>

<td style="text-align:right;">

0.6500125

</td>

<td style="text-align:right;">

0.6518252

</td>

<td style="text-align:right;">

0.8130915

</td>

<td style="text-align:right;">

0.8142746

</td>

</tr>

<tr>

<td style="text-align:left;">

Random Forest

</td>

<td style="text-align:right;">

129.4666012

</td>

<td style="text-align:right;">

0.7805111

</td>

<td style="text-align:right;">

0.8344283

</td>

<td style="text-align:right;">

0.9945771

</td>

<td style="text-align:right;">

0.8868588

</td>

<td style="text-align:right;">

0.9957379

</td>

<td style="text-align:right;">

0.7879086

</td>

<td style="text-align:right;">

0.9934198

</td>

<td style="text-align:right;">

0.8839622

</td>

<td style="text-align:right;">

0.9959796

</td>

</tr>

</tbody>

</table>

# References

<div id="refs" class="references hanging-indent">

<div id="ref-antonio2019hotel">

Antonio, Nuno, Ana de Almeida, and Luis Nunes. 2019. “Hotel Booking
Demand Datasets.” *Data in Brief* 22: 41–49.

</div>

<div id="ref-chen2011search">

Chen, Chih-Chien, Zvi Schwartz, and Patrick Vargas. 2011. “The Search
for the Best Deal: How Hotel Cancellation Policies Affect the Search and
Booking Decisions of Deal-Seeking Customers.” *International Journal of
Hospitality Management* 30 (1): 129–35.

</div>

<div id="ref-numpy">

Harris, Charles R., K. Jarrod Millman, St’efan J. van der Walt, Ralf
Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, et al. 2020.
“Array Programming with NumPy.” *Nature* 585 (7825): 357–62.
<https://doi.org/10.1038/s41586-020-2649-2>.

</div>

<div id="ref-docoptpython">

Keleshev, Vladimir. 2014. *Docopt: Command-Line Interface Description
Language*. <https://github.com/docopt/docopt>.

</div>

<div id="ref-pandaspaper">

McKinney. 2010. “Data Structures for Statistical Computing in Python.”
In *Proceedings of the 9th Python in Science Conference*, edited by
Stéfan van der Walt and Jarrod Millman, 56–61.
[https://doi.org/ 10.25080/Majora-92bf1922-00a](https://doi.org/%2010.25080/Majora-92bf1922-00a%20).

</div>

<div id="ref-scikit-learn">

Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O.
Grisel, M. Blondel, et al. 2011. “Scikit-Learn: Machine Learning in
Python.” *Journal of Machine Learning Research* 12: 2825–30.

</div>

<div id="ref-pandas">

team, The pandas development. 2020. *Pandas-Dev/Pandas: Pandas* (version
latest). Zenodo. <https://doi.org/10.5281/zenodo.3509134>.

</div>

<div id="ref-altair">

VanderPlas, Jacob, Brian Granger, Jeffrey Heer, Dominik Moritz, Kanit
Wongsuphasawat, Arvind Satyanarayan, Eitan Lees, Ilia Timofeev, Ben
Welsh, and Scott Sievert. 2018. “Altair: Interactive Statistical
Visualizations for Python.” *Journal of Open Source Software* 3 (32):
1057.

</div>

<div id="ref-Python">

Van Rossum, Guido, and Fred L. Drake. 2009. *Python 3 Reference Manual*.
Scotts Valley, CA: CreateSpace.

</div>

<div id="ref-xie2007service">

Xie, Jinhong, and Eitan Gerstner. 2007. “Service Escape: Profiting from
Customer Cancellations.” *Marketing Science* 26 (1): 18–30.

</div>

</div>
