# Utilizing Prenatal Data for Early Detection of Pediatric Health Risks: An Exploratory Approach for Improved Clinical Outcomes.

## Motivation
Early identification of diseases and their associated length of hospital stay (LoS) is vital for better treatment options, more effective follow-up arrangements, and lower hospital utilization costs. 

Our research introduces Ped-BERT, a deep learning model developed to predict over 100 diagnosis conditions and the hospital LoS for pediatric patients. Leveraging a confidential dataset of 513.9K mother-baby pairs, including medical diagnosis codes and patient characteristics, Ped-BERT is pre-trained using a masked language modeling (MLM) objective. Fine-tuning enables accurate prediction of primary diagnosis outcomes and LoS based on previous visit history and optionally, maternal health information. We find that Ped-BERT generally outperforms contemporary and state-of-the-art DL classifiers when trained with minimum features. We also find that incorporating mother health attributes leads to significant improvements in model performance overall and across all patient subgroups in our data. 

## Data Collection Process
[1] We collected health data from The California Department of Health Care Access and Information (HCAI), which provides confidential patient-level data sets to researchers eligible through the Information Practices Act (CA Civil Code Section 1798 et seq.). Note that researchers interested in working with this health data should request it directly from [HCAI](https://hcai.ca.gov/data-and-reports/research-data-request-information/) as it is HIPAA protected, and by agreement, we are not allowed to distribute it. 

[2] The geospatial data came from the Census Bureau and includes [2010 ZCTA shapefiles](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=ZIP+Code+Tabulation+Areas_), [2010 county shapefiles](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=Counties+%28and+equivalent%29), [2010 ZCTA to county codes](https://www.census.gov/programs-surveys/geography/technical-documentation/records-layout/2010-zcta-record-layout.html), [ZCTA to zip codes crosswalks](https://github.com/censusreporter/acs-aggregate/blob/master/crosswalks/zip_to_zcta/ZIP_ZCTA_README.md), as well as the [2020 geographical division of California’s 58 counties into ten regions](https://census.ca.gov/regions/). 

## Scripts
We divide our scrips into data cleaning and analysis, as follows: 
[1] The data cleaning folder includes TensorFlow code for preprocessing and merging the health and geospatial datasets;
[2] The analysis folder includes TensorFlow code for conducting the Ped-BERT analysis, encompassing both pre-training and fine-tuning stages. We also include code for logistic regression (LR), random forest (RF), and pre-training and fine-tuning of a transformer decoder-only model (TDecoder). The code presented here supports our downstream analysis of predicting the primary diagnosis and associated hospital LoS for the subsequent medical encounter.

