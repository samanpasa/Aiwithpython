import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def regression_model():
    df = pd.read_csv("weight-height.csv")

    plt.scatter(df["Height"], df["Weight"], alpha=0.5)
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.title("Height vs Weight Scatter Plot")
    plt.show()


    X = df["Height"].values.reshape(-1, 1)
    y = df["Weight"].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)


    plt.scatter(X, y, alpha=0.5, label="Actual Data")
    plt.plot(X, y_pred, color='red', label="Regression Line")
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.title("Regression of Height vs Weight")
    plt.legend()
    plt.show()

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")


regression_model()
