# Calories Burnt Prediction using XGBoost Regressor

This repository contains a project for predicting the number of calories burnt during physical activities using an XGBoost Regressor model. The project focuses on analyzing various features like age, gender, body temperature and heart rate, and building a regression model to predict calories burnt based on these features.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)

## Introduction

Predicting the number of calories burnt is an important task for health and fitness analysis. This project aims to build a machine learning model to predict the number of calories burnt based on various physical and personal characteristics.

## Dataset

The dataset used in this project is from Kaggle’s Calories Burnt Dataset. It contains information about age, gender, heart rate, body temperature and calories burnt during physical activities. The dataset is split into training and testing sets to evaluate the model's performance.

- [Calories Burnt Dataset on Kaggle](https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos)

## Installation

To run this project, you need to have Python installed on your machine. You can install the required dependencies using `pip`.

```
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

Requirements
Python 3.x
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
XGBoost

## Usage

1. Clone the repository to your local machine:

```
   git clone https://github.com/srijosh/Calories-Burnt-Prediction-using-XGBoost-Regressor.git

```

2. Navigate to the project directory:
   cd Calories-Burnt-Prediction-using-XGBoost-Regressor

3. Download the dataset from Kaggle and place it in the project folder

4. Open and run the Jupyter Notebook:
   jupyter notebook Calories_Burnt_Prediction.ipynb

## Model

The model used in this project is an XGBoost Regressor. The data is preprocessed by encoding categorical variables and splitting the dataset into training and testing sets. Key steps include:

## Data Preprocessing

Label Encoding: Converting categorical variables (e.g., gender) to numeric codes.
Train-Test Split: Splitting the dataset into training and testing sets to evaluate the model's performance.

## Model Training

XGBoost Regressor: An gradient boosting algorithm is used to train the model and make predictions.

### Evaluation

The model is evaluated using the following metric:

- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in the predictions without considering their direction. It gives an idea of how far off the predictions are from the actual calorie values, with lower values indicating better model performance.
