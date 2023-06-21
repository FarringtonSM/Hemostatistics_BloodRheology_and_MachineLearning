# GaussianProcessRegression

***The First PLace to Start***

"UseModel.py"

This python script is used to make model predictions that are present in the paper. You can predict a single set of Casson parameters or make a surface plot with a range of hematocrit and fibrinogen.


"Main_CreateModel.py"

Python script used to create the Gaussian process regression model used for the results in the paper.

"GPR_testingModel.py"

Python script to generate a training/testing split of the blood rheology data and test with Gaussian process regression

"GPR_validationKFold.py"

Python script to validate the model using the standard technique of K-fold validation.

"MetricsAndFigures.py"

Python script to use the Guassian process regression model saved as .pkl files to create the figures and metrics presented in the paper.
