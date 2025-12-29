Maine Crop Production & Weather Analysis

**Project Overview**

This project investigates the relationship between agricultural crop production and weekly weather conditions in the state of Maine. Using publicly available USDA datasets, we collected and analyzed production and weather data, then applied machine-learning models to predict crop production.

The workflow integrates:

API data collection
PDF web scraping
Data cleaning and feature engineering
Visualization
Machine learning (regression & random forest)

**Data Sources**
1. Crop Production Data

Source: USDA NASS QuickStats API
Scope: County-level crop production data for Maine
Years: 2007, 2012, 2017, 2022

2. Weather Data

Source: USDA Crop Progress & Condition weekly PDF reports
Variables extracted:
Temperature (high, low, average)
Weekly precipitation (inches)
Number of rain days
Years: 2007, 2012, 2017, 2022


**Analysis & Modeling**
I evaluated two predictive models:

Ridge Regression
Serves as a linear baseline
Helps identify relationships between weather variables and production

Random Forest Regression
Captures nonlinear interactions
Provides improved predictive performance and feature importance insights (only with larger datasets).

Model performance was evaluated using MSE, R^2, MSR ,validation datasets and visualized with comparison plots.

**Key Takeaways**

The dataset was too small to build more accurate models. Using these models on much larger and varied datasets produces more accurate models . Overall, hot and dry climate corresponded with higher crop production in Maine. 

Ridge Regression models outperform random forest in this scenario due to the nature of the dataset.
