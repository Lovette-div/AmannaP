# Agricultural Market Price Predictor

## Project Overview

This project implements a machine learning model to predict agricultural product's market prices based on various features such as produce variety, commodity type, packaging details, and date. The model is deployed as a web application using Flask, allowing users to input details and receive price predictions.

## Features

- Predicts agricultural market prices based on multiple input features
- Web interface for easy user interaction
- Handles both numerical and categorical input data
- Uses machine learning techniques including feature scaling and one-hot encoding

## Technologies Used

- Python 3.x
- Flask (Web Framework)
- Scikit-learn (Machine Learning Library)
- NumPy (Numerical Computing Library)
- Pandas (Data Manipulation Library)
- Joblib (Model Persistence)
- pyngrok (For exposing local server to the internet during development)

## Setup and Installation

1. Clone the repository:
2. Install required packages: like flask-ngrok pyngrok and requests.

3. Ensure you have the trained model file (`base_model.sav`) in the project directory.

## Running the Application

1. Start the Flask application:
   2. Access the web interface through the URL provided in the console output.

## Usage

1. Navigate to the web interface.
2. Fill in the required information in the form:
- Produce Variety
- Commodity Type
- Unit
- Date
- Package Weight in Kg
3. Submit the form to receive a price prediction.
  ## Model Training

The `Final.py` and 'Final.ipynb' script contains the code used to preprocess the data, train the model, and save it. Key steps include:

- Loading and cleaning the dataset
- Feature engineering (including datetime processing)
- Splitting data into training and testing sets
- Scaling features using StandardScaler
- Training the model (We trained Regressors like: DecisionTree, RandomForest Regressor and Linear Regressor, we fine-tuned the model using GridSearchCV)
- We passed the model through Ensemble learning models like GradientBoost, XGBoost, Stacking and Voting.
- We went ahead to explore other deep learning models like CNN and LSTM.
- Saving the best performing trained model(XGBoost) using joblib

## File Structure

- `app.py`: Main Flask application
- `Final.py` and 'Final.ipynb': Script for training and saving the model
- `base_model.sav`: Saved trained model
- `templates/`: Directory containing HTML templates (if any)
- ## Future Improvements

- Implement user authentication
- Add more advanced feature engineering
- Explore and compare different machine learning models
- Implement periodic model retraining with new data

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## Contact

Chika Perpetual Amanna - chika.amanna@ashesi.edu.gh
Lovette Carthy Philips - lovette.philips@ashesi.edu.gh

Project Youtube Link: https://www.youtube.com/watch?v=_izmRT6qnFw

