#import necessary libraries
import numpy as np  # NumPy for numerical operations
import pandas as pd  # Pandas for handling data (though it's not used in this script)
import csv  # CSV module to read data from the file

# Define the file name containing advertising data
file_name = "./Advertising.csv"

# Initialize empty numpy arrays for each column in the dataset
tv = np.array([])  # Array for TV advertising budget
radio = np.array([])  # Array for Radio advertising budget
newsp = np.array([])  # Array for Newspaper advertising budget
sales = np.array([])  # Array for Sales data

# Open the CSV file and process the data into arrays
with open(file_name, mode='r') as file:
    csv_reader = csv.reader(file)  # Read the file as a CSV
    header = next(csv_reader)  # Skip the header row
    for row in csv_reader:  # Iterate through each row in the CSV file
        tv = np.append(tv, float(row[1]))  # Append TV budget values
        radio = np.append(radio, float(row[2]))  # Append Radio budget values
        newsp = np.append(newsp, float(row[3]))  # Append Newspaper budget values
        sales = np.append(sales, float(row[4]))  # Append Sales values

# Ensure all arrays have the same length
length = len(tv)  # Get the number of data points
assert len(tv) == len(radio) == len(newsp) == len(sales)  # Check if all arrays are of equal length


# Function to compute linear regression statistics for a given independent variable (x)
def function(x):
    x_ = np.mean(x)  # Compute the mean of x (independent variable)
    sales_ = np.mean(sales)  # Compute the mean of sales (dependent variable)

    # Compute beta1 (slope) using the least squares formula
    beta1 = (np.sum((x - x_) * (sales - sales_))) / (np.sum((x - x_)**2))
    beta1 = round(beta1, 5)  # Round to 5 decimal places

    # Compute beta0 (intercept)
    beta0 = sales_ - beta1 * x_
    beta0 = round(beta0, 5)  # Round to 5 decimal places

    # Predict sales based on the regression line
    sales_pred = beta0 + x * beta1

    # Compute the Residual Sum of Squares (RSS)
    RSS = np.sum((sales - sales_pred) ** 2)

    # Compute the Residual Standard Error (RSE)
    RSE = np.sqrt((1 / (length - 2)) * RSS)

    # Compute the Total Sum of Squares (TSS)
    TSS = np.sum((sales - sales_) ** 2)

    # Compute R-squared (R²) to measure goodness of fit
    R2 = (TSS - RSS) / TSS

    # Compute F-Statistic for model significance testing
    p = 1  # Number of predictor variables (since we're using simple linear regression)
    FS = ((TSS - RSS) / p) / (RSS / (length - p - 1))

    # Print the calculated statistics
    print("RSS", RSS, "\nRSE", RSE, "\nTSS", TSS, "\nR2", R2, "\nFS", FS)


# Run the function for each independent variable (TV, Radio, Newspaper)
if __name__ == "__main__":
    function(tv)  # Compute regression for TV advertising
    function(radio)  # Compute regression for Radio advertising
    function(newsp)  # Compute regression for Newspaper advertising
