# Weather Forecasting and Model Evaluation

## Overview
This project involves training, evaluating, and comparing machine learning models for weather forecasting. The models predict daily precipitation based on other weather parameters such as Specific Humidity, Relative Humidity, and Temperature.

## Features
The key features of the dataset include:

- **Specific Humidity**
- **Relative Humidity**
- **Temperature**
- **Precipitation**
  
![existing_data](https://github.com/user-attachments/assets/3f2f895a-d383-4892-b782-4473ab496e9b)

The project utilizes these features for preprocessing, model training, and prediction tasks.

## Methodology
1. **Data Preprocessing**:
    - Missing values were filled using column-wise means.
    - The data was scaled using MinMaxScaler.
    - Date information was parsed and set as the index.

2. **Models Used**:
    - Random Forest Regressor
    - Gradient Boosting Regressor
    - Support Vector Regressor (SVR)
    - Long Short-Term Memory (LSTM) Neural Network

3. **Hyperparameter Optimization**:
    - Hyperparameters were optimized using GridSearchCV for Random Forest, Gradient Boosting, and SVR models.

4. **Cross-Validation**:
    - TimeSeriesSplit with 5 folds was used for model evaluation.
    - Metrics calculated include Root Mean Squared Error (RMSE) and R-squared (R²).

5. **Visualization**:
    - Time-series plots and scatter plots were created for analysis.
    - Comparison plots for RMSE and R² across all models.

6. **Forecasting**:
    - The best-performing model was used to forecast precipitation for new inputs.

## Results
The performance of the models was evaluated using the following metrics:
- Train RMSE
- Test RMSE
- Train R²
- Test R²

The model with the highest Test R² was selected as the best model and saved.

## Repository Structure
```
Output_03/
├── input/
│   └── 5year_forecast_combined.csv
├── models/
│   └── best_model.keras or best_model.joblib
├── plots/
│   ├── existing_data.png
│   ├── scatter_cross_validation.png
│   └── Model_Comparison.png
├── results/
│   └── comparison_metrics.csv
└── Weather_Data.csv
```

## How to Run
1. Clone the repository and ensure you have Python installed.
2. Install the necessary libraries using:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script to preprocess data, train models, and generate plots:
   ```bash
   python main.py
   ```
4. View the results in the `Output_03` directory.

## Dependencies
- Python 3.8+
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow/Keras
- Joblib

## Acknowledgments
This project utilizes publicly available weather datasets. Special thanks to the creators of the libraries and tools used in this project:

- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Matplotlib](https://matplotlib.org/)

## License
This project is licensed under the Creative Commons Zero v1.0 Universal (CC0 1.0). You are free to use, modify, and distribute the code and documentation without restrictions.

## Contact
For questions, feedback, or contributions, contact:

- Name: Rod Will
- Email: rhudwill@gmail.com

## References
1. Scikit-learn Documentation: https://scikit-learn.org/stable/documentation.html
2. TensorFlow/Keras Documentation: https://www.tensorflow.org/api_docs
3. Matplotlib Documentation: https://matplotlib.org/stable/contents.html

