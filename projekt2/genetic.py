import pandas as pd
import matplotlib as mpl
import pygad
from datetime import datetime, timedelta
from preprocessing import predictLSTM

mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams["figure.figsize"] = (10, 7)

data = pd.read_csv("resources/stock_data.csv")
df = data.copy()

end_date = datetime.today()
start_date = end_date - timedelta(days=30)

df["Date"] = pd.to_datetime(df["Date"])

df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

if df.empty:
    raise ValueError("No data in the selected date range. Check if 'start_date' and 'end_date' are correct.")

df_long = df.melt(id_vars=["Date"], var_name="ticker", value_name="price")
df_long = df_long.rename(columns={"Date": "date"})

tickers = df_long["ticker"].unique()
tickers_map = {ticker: idx for idx, ticker in enumerate(tickers)}
tickers_map_reverse =  {v: k for k, v in tickers_map.items()}

df_long["ticker_index"] = df_long["ticker"].map(tickers_map)
first_prices = df_long.groupby("ticker")["price"].transform("first")
df_long["adj_price"] = df_long["price"] / first_prices

df_final = df_long[["ticker", "ticker_index", "date", "adj_price"]]


# Sprawdzenie czy ilość danych jest równa (spółka nie dołączyła/odeszła z indeksu w ostatnim czasie)
print(df_final.groupby("ticker").count().sort_values("date"))

# df = df[df["ticker"] != "OGN"]

#!tomorrow_returns = predictLSTM(df)
#print(tomorrow_returns)



def get_portfolio_tickers(df, tickers):
    portfolio = df[df['ticker_index'].isin(tickers)]
    unique_tickers = portfolio['ticker'].unique()
    return unique_tickers


def portfolio_LSTM_return(portfolio_tickers) -> float:
    sum_returns = 0.0
    for ticker in portfolio_tickers:
        sum_returns += tomorrow_returns[ticker]
    return sum_returns / len(portfolio_tickers)


def portfolio_generate(df, tickers):
    portfolio = df[df['ticker_index'].isin(tickers)]
    portfolio = portfolio.groupby("date", as_index=False).sum()
    portfolio = portfolio.sort_values("date")
    return portfolio, get_portfolio_tickers(df, tickers)

def get_portfolio_tickers(df_final, tickers):
    portfolio = df_final[df_final['ticker_index'].isin(tickers)]  # Changed df to df_final
    unique_tickers = portfolio['ticker'].unique()
    return unique_tickers

def portfolio_history_return(portfolio):
    first_price = portfolio["adj_price"].iloc[0]
    last_price = portfolio["adj_price"].iloc[-1]
    return last_price / first_price - 1

def portfolio_risk(portfolio):
    portfolio["daily_change"] = portfolio["adj_price"].diff(1)
    portfolio["daily_change"] = portfolio["daily_change"] / portfolio["adj_price"]
    return portfolio["daily_change"].std()

def fitness_func(ga_instance, solution, solution_idx):
    portfolio, portfolio_tickers = portfolio_generate(df_final, solution)  # Changed df to df_final
    ret = portfolio_history_return(portfolio)
    ris = portfolio_risk(portfolio)
    #! expected_return_points = (portfolio_LSTM_return(portfolio_tickers) - 1) * 2000
    #! fitness = (ret / ris) - expected_return_points
    fitness = (ret / ris)

    return fitness

def visualize(df_final, solution) -> None:  # Changed parameter name
    solution_fitness = fitness_func(None, solution, None)
    portfolio, _ = portfolio_generate(df_final, solution)  # Changed df to df_final
    portfolio["adj_price"] = (portfolio["adj_price"] / portfolio["adj_price"].iloc[0]) * 100
    ax = portfolio.plot.line(x="date", y="adj_price")
    ax.set_ylim(90, 190)
    ret = round(portfolio_history_return(portfolio) * 100, 1)
    ris = round(portfolio_risk(portfolio) * 100, 1)
    print("Tickers map:", tickers_map)
    print("Tickers map reversed", tickers_map_reverse)
    print(f"Parameters of the best solution : {[tickers_map_reverse[i] for i in solution]}")
    print(f"Return: {ret}%")
    print(f"Risk: {ris}%")
    solution_fitness_scalar = solution_fitness.item()
    print(f"Risk adjusted return = {round(solution_fitness_scalar, 1)}%")


ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=50,
                       fitness_func=fitness_func,
                       sol_per_pop=90,
                       num_genes=10,
                       init_range_low=0,
                       init_range_high=len(tickers)-1,
                       parent_selection_type="sss",
                       keep_parents=30,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=30,
                       gene_type=int,
                       allow_duplicate_genes=True,
                       random_seed=2,
                       mutation_by_replacement=False)
ga_instance.run()


[solution, solution_fintess, __] = ga_instance.best_solution()

if len(solution) > 0:
    visualize(df_final, solution)
    print(solution_fintess)
else:
    print("Warning: Empty solution - no tickers selected")


for i, j in zip(ga_instance.best_solutions, ga_instance.best_solutions_fitness):
    print([(tickers_map_reverse[k], k) for k in sorted(i)], j)

ga_instance.plot_fitness(save_dir="docs/plots/genetic_learning_result.png")
