import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('datasets/Ecommerce Customers')
print(df.head()) # Display the first few rows of the dataframe



df.info() # Get information about the dataframe
print(df.describe()) # Get statistical summary of the dataframe


### Exploratory Data Analysis (EDA)

# Jointplot of Time on Website vs Yearly Amount Spent(Overall relationship between these two columns)
sns.jointplot(x='Time on Website', 
              y='Yearly Amount Spent', 
              data=df) # Scatter plot of Time on Website vs Yearly Amount Spent
plt.show()

# Jointplot of Time on App vs Yearly Amount Spent(Overall relationship between these two columns)
sns.jointplot(x='Time on App', 
              y='Yearly Amount Spent', 
              data=df) # Scatter plot of Time on App vs Yearly Amount Spent
plt.show() # from the plot we can see that there is a stronger correlation between Time on App and Yearly Amount Spent


### Pairplot all numerical values
sns.pairplot(df , 
             kind='scatter', 
             plot_kws={'alpha':0.5}) # Pairplot 
plt.show()  


### Linear Model Plot of Length of Membership vs Yearly Amount Spent
sns.lmplot(x='Length of Membership',
           y='Yearly Amount Spent', 
           data=df,
           scatter_kws={'alpha':0.3}
           
           ) # Linear regression plot
plt.show() # from the plot we can see that Length of Membership is the most correlated feature with Yearly Amount Spent



from sklearn.model_selection import train_test_split


X = df[['Time on App', 'Time on Website', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']       # Here X is in uppercase because it is a matrix (2D array) and y is in lowercase because it is a vector (1D array)      

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42
                                                    ) # Split the data into training and testing sets       




###Training the Model
from sklearn.linear_model import LinearRegression

lm = LinearRegression() # Create a Linear Regression model object
lm.fit(X_train, y_train) # Fit the model to the training data
print('Coefficients: \n', lm.coef_) # Print the coefficients of the model
                                    # Coefficients represent the change in the target variable for a one unit change in the predictor variables, holding all other predictors constant.
                                    # For example, a coefficient of 61.27 for 'Time on App' means that for each additional unit of time spent on the app, the yearly amount spent increases by approximately $61.27, assuming all other factors remain constant.



cdf = pd.DataFrame(lm.coef_,
                   X.columns,  
                   columns=['Coefficient']) # Create a dataframe to display the coefficients
print(cdf) # Print the coefficients dataframe





### Predictions from our Model
predictions = lm.predict(X_test) # Predict the target variable for the test data
plt.scatter(y_test, 
            predictions) # Scatter plot of actual vs predicted values
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predictions')
plt.show()



### Evaluating the Model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math


print('Mean Absolute Error:', mean_absolute_error(y_test, predictions)) # Mean abisolute error is the average of the absolute errors between the predicted and actual values
print('Mean Squared Error:', mean_squared_error(y_test, predictions)) # Mean squared error is the average of the squared errors between the predicted and actual values
print('Root Mean Squared Error:', math.sqrt(mean_squared_error(y_test, predictions))) # Root mean squared error is the square root of the mean squared error
     

### Residuals
residuals = y_test - predictions # Calculate the residuals (errors) between the actual and predicted values
sns.histplot(residuals, 
             bins=20, 
             kde=True) # Plot the distribution of the residuals
plt.xlabel('Residuals')
plt.show() # The residuals should be normally distributed around zero, indicating that the model's predictions are unbiased. 











