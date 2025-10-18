import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    return pd, plt


@app.cell
def _(pd):
    df = pd.read_csv('sales_data.csv')
    df
    return (df,)


@app.cell
def _(df):
    X = df['Day']
    y = df['Revenue']
    return X, y


@app.cell
def _(X, plt, y):
    plt.figure(figsize=(10, 6))
    plt.plot(X, y)
    plt.xlabel('Day')
    plt.ylabel('Revenue')
    plt.title('Revenue vs Day')
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(df, pd):
    days = pd.to_datetime(df["Day"])  # make sure it's datetime
    monthly_mean = df.groupby(days.dt.to_period("M"))["Revenue"].mean()
    monthly_mean
    return


@app.cell
def _(df, pd, plt):
    df["Day"] = pd.to_datetime(df["Day"])
    monthly_means = df.resample("ME", on="Day")["Revenue"].mean()
    monthly_means = monthly_means[monthly_means != 0]

    monthly_means.plot(kind="line", marker="o")  # or kind="bar"
    plt.xlabel("Month")
    plt.ylabel("Average Value")
    plt.title("Average Value per Month")
    plt.show()
    return


@app.cell
def _(df, pd, plt):
    df['Year'] = df['Day'].dt.year  # assuming 'Day' is a datetime column
    # Sum revenue per month per year
    monthly_sum_per_year = df.groupby(['Year', 'Month'])['Revenue'].sum()
    # Average revenue per month across the years
    monthly_mean_across_years = monthly_sum_per_year.groupby('Month').mean() #* 1.4

    df["MonthName"] = df["Day"].dt.strftime("%b")
    order = [9,10,11,12,1,2,3,4,5,6,7,8]   # Sep → Aug

    monthly_mean_across_years = monthly_mean_across_years.reindex(order)
    month_labels = [pd.to_datetime(str(m), format="%m").strftime("%b") for m in order]
    year_sum = monthly_mean_across_years.sum()

    monthly_mean_across_years = monthly_mean_across_years / 1e6
    monthly_mean_across_years.plot(kind="bar")
    plt.xticks(range(len(order)), month_labels, rotation=45)
    plt.xlabel("Month")
    plt.ylabel("DKK (Millions)")
    plt.title(f"Ticket Sale Forecast of Next Year\nSum: {round(year_sum / 1e6, 2)} DKK Million")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(monthly_meanss):
    monthly_meanss.sum()
    return


if __name__ == "__main__":
    app.run()
