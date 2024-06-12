import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pygad
import numpy as np
import itertools
import random
from preprocessing import predictLSTM

mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams["figure.figsize"] = (10, 7)

data = pd.read_csv("resources/new_data.csv")
df = data.copy()
tomorrow_returns = predictLSTM(df)
# start_date = "2021-04-01"
# end_date = "2022-03-01"
# df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
df = df[["ticker", "date", "Close"]].rename(columns={"Close": "price"}).reset_index(drop=True)
tickers = df["ticker"].unique()
tickers_map = {i: j for i, j in zip(tickers, range(len(tickers)))}
tickers_map_reverse = {j: i for i, j in zip(tickers, range(len(tickers)))}

df["ticker_index"] = df["ticker"].map(tickers_map)
firsts = (df.groupby('ticker').transform('first'))
df["adj_price"] = df["price"] / firsts["price"]
df = df[["ticker", "ticker_index", "date", "adj_price"]]

## Drop OGN as it joined SP500 midway
df = df[df["ticker"] != "OGN"]

df.groupby("ticker").count().sort_values("date")


## Define fitness function and utils

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
    # print("Tickers in the portfolio:", get_portfolio_tickers(df, tickers))  # Wyświetlanie tickerów
    return portfolio, get_portfolio_tickers(df, tickers)


def portfolio_history_return(portfolio):
    first_price = portfolio["adj_price"].iloc[0]
    last_price = portfolio["adj_price"].iloc[-1]
    return last_price / first_price - 1


def portfolio_risk(portfolio):  # * odchylenie standardowe dziennych zmian
    portfolio["daily_change"] = portfolio["adj_price"].diff(1)
    portfolio["daily_change"] = portfolio["daily_change"] / portfolio["adj_price"]
    return portfolio["daily_change"].std()


def fitness_func(ga_instance, solution, solution_idx):
    portfolio, portfolio_tickers = portfolio_generate(df, solution)
    ret = portfolio_history_return(portfolio)
    ris = portfolio_risk(portfolio)
    expected_return = portfolio_LSTM_return(portfolio_tickers) * 1000 - 1000
    fitness = (ret / ris) + expected_return
    return fitness


def visualize(df, solution) -> None:
    solution_fitness = fitness_func(None, solution, None)
    portfolio, _ = portfolio_generate(df, solution)
    portfolio["adj_price"] = (portfolio["adj_price"] / portfolio["adj_price"].iloc[0]) * 100
    ax = portfolio.plot.line(x="date", y="adj_price")
    ax.set_ylim(90, 190)
    ret = round(portfolio_history_return(portfolio) * 100, 1)
    ris = round(portfolio_risk(portfolio) * 100, 1)

    print(f"Parameters of the best solution : {[tickers_map_reverse[i] for i in solution]}")
    print(f"Return: {ret}%")
    print(f"Risk: {ris}%")
    solution_fitness_scalar = solution_fitness.item()
    print(f"Risk adjusted return = {round(solution_fitness_scalar, 1)}%")


## Define Genetic Algorithm

fitness_function = fitness_func
num_generations = 60
num_genes = 10

sol_per_pop = 90
num_parents_mating = 50

init_range_low = 100
init_range_high = 497
gene_type = int

parent_selection_type = "sss"
keep_parents = 30

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 30
mutation_by_replacement = False

## Initiate and run genetic algorithm

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       gene_type=gene_type,
                       #    allow_duplicate_genes=False,
                       random_seed=2,
                       mutation_by_replacement=mutation_by_replacement)
ga_instance.run()
for i, j in zip(ga_instance.best_solutions, ga_instance.best_solutions_fitness):
    print([(tickers_map_reverse[k], k) for k in sorted(i)], j)

## Plot training, best results
ga_instance.plot_fitness(save_dir="docs/learning_result.png")
[solution, _, __] = ga_instance.best_solution()
visualize(df, solution)

# 10 spółek o największym zysku, nie uwzględniając ryzyka
firsts = df.groupby("ticker_index", as_index=False).first()
firsts = firsts.rename({"adj_price": "first_price"}, axis=1)[["ticker_index", "first_price"]]
lasts = df.groupby("ticker_index", as_index=False).last()
lasts = lasts.rename({"adj_price": "last_price"}, axis=1)[["ticker_index", "last_price"]]

df_ = firsts.merge(lasts, on="ticker_index", how="left")
df_["return"] = df_["last_price"] / df_["first_price"]
df_ = df_.sort_values("return", ascending=False)
best_return = df_.head(10)["ticker_index"].unique()

visualize(df, best_return)

## S&P 500 benchmark

visualize(df, df["ticker_index"].unique())
