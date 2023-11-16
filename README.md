# HomeCreditDefaultRisk

## The Business problem

This is a binary Classification task: I want to predict whether the person applying for a home credit will be able to repay their debt or not. The model will have to predict a 1 indicating the client will have payment difficulties: he/she will have late payment of more than X days on at least one of the first Y installments of the loan in our sample, 0 in all other cases.

I will use [Area Under the ROC Curve](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=es_419) as the evaluation metric, so my models will have to return the probabilities that a loan is not paid for each input data.

## Technical aspects

The technologies involved are:
- Python as the main programming language
- Pandas for consuming data from CSVs files
- Scikit-learn for building features and training ML models
- Matplotlib and Seaborn for the visualizations
- Jupyter notebooks to make the experimentation in an interactive way
