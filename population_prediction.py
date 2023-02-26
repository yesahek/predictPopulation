import sys

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# model instantiating an instance of the LogisticRegression object
model = LinearRegression()


def modeling():
    # accepting csv file and test size from the user:
    csv_data = input("Enter the csv file name with the extension : ")
    size = float(input("Enter test size '0.2' is suggested : "))

    data = pd.read_csv(csv_data)
    print("\nReading done")
    # Split the dataset into features(x) and target(y)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=0)

    # Train the model
    model.fit(x_train, y_train)

    # Let's Evaluate the model
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n*----Evaluation of the model----*")
    print("\nMean Squared Error :", mse)
    print("\nR-squared:  ", r2)

    # Output the predicted values
    # print("\nPredicted values:", predictions)

    # changing predictions and x_test variable datatype from numpy.ndarray in to list datatype
    predictions = [int(i) for i in predictions]
    x_test = [int(i) for i in x_test]
    # df = pd.DataFrame(predictions, y_test)
    df = pd.DataFrame({'year': x_test, 'test': y_test, 'predictions': predictions})

    print("\n*---- Evaluation Result----*\n")
    print(df)


def predictor():
    year = int(input("Enter a Year : "))
    population = model.predict([[year]])
    print('\nThe predicted population in', year, 'is', float(population))


modeling()
ask = input("\ndo you want to try. \n Enter 'y' for Yes or Enter any key to Exit! : ")
if 'y' == ask or ask == 'Y':
    predictor()
else:
    sys.exit()
