House Price ML
Small machine learning project for predicting real estate prices using scikit-learn.

Dataset
Real Estate Valuation dataset with 414 observations.

Features used:

House age
Distance to the nearest MRT station
Number of convenience stores
Latitude
Longitude
Target:

House price per unit area
Models
Model	Test MAE	Test R²
Dummy Regressor	10.92	-0.025
Linear Regression	5.35	0.675
Decision Tree	4.73	0.754
Random Forest	4.25	0.794
Random Forest achieved the best test performance.

Setup
Create a virtual environment and install the dependencies:

pip install -r requirements.txt

Run the project:

python main.py