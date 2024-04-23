import pandas as pd
from apyori import apriori

def main(filename: str):
    data = pd.read_csv(filename)

    transactions = []
    for index, row in data.iterrows():
        transaction = []
        for col in data.columns:
            transaction.append(str(row[col]))
        transactions.append(transaction)

    result = apriori(transactions, min_support=0.2, min_confidence=0.8, min_lift=1.2, min_length=2)

    for rule in result:
        print("Reguła: " + str(rule.items))
        print("Wsparcie: " + str(rule.support))
        print("Ufność: " + str(rule.ordered_statistics[0].confidence))
        print("Lift: " + str(rule.ordered_statistics[0].lift))
        print("------------------------------")


if __name__ == '__main__':
    main("titanic.csv")


# confidence - prawdopodobieństwo że rekordy spełniające regółę X spełniają też Y
# lift - określa jak bardzo występienie jednego zdarzenia zwiększa szanse wystąpienia 2
# support - miara częśtości występowania regóły w zbiorze danych
