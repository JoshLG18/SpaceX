Project Overview
This project involves analyzing and predicting SpaceX mission outcomes based on historical launch data. Using Python, SQL, and various machine learning models, we explore factors affecting launch success rates, such as launch sites, payload mass, orbit type, and more. The aim is to provide actionable insights that help understand trends and reliability factors in SpaceX's launch operations.

Data Collection

    API Access: Data is gathered from the SpaceX API to retrieve information on launches, rockets, and payloads.
    Web Scraping: Supplementary data on launch sites and other features were collected using web scraping techniques.
    Data Wrangling: The data is cleaned, transformed, and prepared for analysis. This includes handling missing values, standardizing formats, and merging datasets.


Exploratory Data Analysis (EDA)

    The analysis includes:
    Visualizing success rates by launch site, orbit, and payload mass.
    Exploring trends in launch outcomes over time.
    Creating interactive maps with Folium to display launch sites, proximity to infrastructure, and distances to nearby features.

Predictive Modeling

    Machine learning models are used to predict the likelihood of launch success based on various factors. The models employed include:
    Logistic Regression
    Decision Trees
    Support Vector Machines (SVM)
    K-Nearest Neighbors (KNN)
    Accuracy: Each model was tuned using GridSearchCV to achieve an accuracy exceeding 83%.

