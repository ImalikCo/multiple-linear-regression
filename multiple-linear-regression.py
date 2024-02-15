import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('.csv')

X = data.drop('column_x', axis=1)  # Columns you want to 'train' with
y = data['column_y']  # Column you want to 'predict'

# Splitting data into training & testing sets, change test_size and random_state to your own wish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate the MSE
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

coefficients = model.coef_
intercept = model.intercept_

print('Change in target variable per unit:', coefficients)
print('Predicted value of target:', intercept)