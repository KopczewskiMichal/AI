from datetime import date
import math

class Main:
  def __init__(self, name, birthDate) -> None:
    self.feelingTypes =  {
      "yp": 23,
      "ye": 28,
      "yi": 33
    }
    self.name = name
    self.birtDate = birthDate
    self.daysOfLife = (date.today() - self.birtDate).days
    self.countFeelings()


  @staticmethod
  def askForData ():  
    name = input("Input your name: ")
    dayOfBirth = input("Input your day of birth: ")
    monthOfBirth = input("Input your month of birth: ")
    yearOfBirth = input("Input your year of birth: ")
    birthDateTime = date(int(yearOfBirth), int(monthOfBirth), int(dayOfBirth))
    print(f"Witaj {name}\nUrodziłeś się {birthDateTime}")
    return Main(name, birthDateTime)
  
  def countFeelings(self):
    def countSingleFeeling(type):
      def countSingleFeelingHelper(daysCount, n):
        days = daysCount % n
        return math.sin(2 * math.pi * days / n)

      todayFeeling = countSingleFeelingHelper(self.daysOfLife, self.feelingTypes[type])
      print(f"\nDzisiaj Twoje {type} wynosi: {todayFeeling}")
      if todayFeeling < - 0.5:
        tomorrowFeeling = countSingleFeelingHelper(self.daysOfLife + 1, self.feelingTypes[type])
        print(f"Współczujemy i życzymy miłego dnia\nJutro będzie {type} wynosiło {tomorrowFeeling}")
      if todayFeeling > 0.5:
        tomorrowFeeling = countSingleFeelingHelper(self.daysOfLife + 1, self.feelingTypes[type])
        print(f"Gratulacje!\nJutro będzie {type} wynosiło {tomorrowFeeling}")


    countSingleFeeling("yp")
    countSingleFeeling("ye")
    countSingleFeeling("yi")
    

if __name__ == "__main__":
  myObj = Main.askForData();

# Powyższy kod napisałem samodzielnie, zajęło to ok. 50 min

# Poniżej wersja popraweiona przez chat gpt w wersji darmowej w czasie ok. 30 sekund
# Kod również działa
# Zmiany: poprawa nazewnictwa, import math w innym miejscu, lepszy wydruk daty

from datetime import date
import math

class Main2:
    def __init__(self, name, birth_date) -> None:
        self.feeling_types =  {
            "yp": 23,
            "ye": 28,
            "yi": 33
        }
        self.name = name
        self.birth_date = birth_date
        self.days_of_life = (date.today() - self.birth_date).days
        self.count_feelings()

    @staticmethod
    def ask_for_data ():  
        name = input("Input your name: ")
        day_of_birth = input("Input your day of birth: ")
        month_of_birth = input("Input your month of birth: ")
        year_of_birth = input("Input your year of birth: ")
        birth_date = date(int(year_of_birth), int(month_of_birth), int(day_of_birth))
        print(f"Witaj {name}\nUrodziłeś się {birth_date.strftime('%d-%m-%Y')}")
        return Main(name, birth_date)
  
    def count_feelings(self):
        def count_single_feeling(type):
            def count_single_feeling_helper(days_count, n):
                days = days_count % n
                return math.sin(2 * math.pi * days / n)

            today_feeling = count_single_feeling_helper(self.days_of_life, self.feeling_types[type])
            print(f"\nDzisiaj Twoje {type} wynosi: {today_feeling}")
            if today_feeling < -0.5:
                tomorrow_feeling = count_single_feeling_helper(self.days_of_life + 1, self.feeling_types[type])
                print(f"Współczujemy i życzymy miłego dnia\nJutro będzie {type} wynosiło {tomorrow_feeling}")
            if today_feeling > 0.5:
                tomorrow_feeling = count_single_feeling_helper(self.days_of_life + 1, self.feeling_types[type])
                print(f"Gratulacje!\nJutro będzie {type} wynosiło {tomorrow_feeling}")

        count_single_feeling("yp")
        count_single_feeling("ye")
        count_single_feeling("yi")

if __name__ == "__main__":
    Main2.ask_for_data()

# Poniżej kod napsany wyłącznie przez AI

import datetime
import math

# Funkcja obliczająca wartość biorytmu dla danej fali i dnia
def calculate_biorhythm(day, cycle):
    return math.sin(2 * math.pi * day / cycle)

# Funkcja obliczająca dzień życia użytkownika na podstawie daty urodzenia
def calculate_days_alive(birth_date, current_date):
    delta = current_date - birth_date
    return delta.days

# Funkcja wyświetlająca komunikat na podstawie wartości biorytmu
def display_message(biorhythm_type, value):
    if value >= 0.5:
        print(f"Twój {biorhythm_type} wynosi {value:.2f}. Czujesz się świetnie!")
    elif value <= -0.5:
        print(f"Twój {biorhythm_type} wynosi {value:.2f}. To może być trudny dzień, ale nie poddawaj się!")
    else:
        print(f"Twój {biorhythm_type} wynosi {value:.2f}. Czujesz się w miarę dobrze.")

# Funkcja główna programu
def main():
    # Pobranie danych od użytkownika
    name = input("Podaj swoje imię: ")
    year = int(input("Podaj rok urodzenia (np. 1990): "))
    month = int(input("Podaj miesiąc urodzenia (np. 1 dla stycznia): "))
    day = int(input("Podaj dzień urodzenia: "))

    # Obliczenie daty urodzenia i obecnej daty
    birth_date = datetime.datetime(year, month, day)
    current_date = datetime.datetime.now()
    days_alive = calculate_days_alive(birth_date, current_date)

    # Obliczenie wartości biorytmów
    physical_biorhythm = calculate_biorhythm(days_alive, 23)
    emotional_biorhythm = calculate_biorhythm(days_alive, 28)
    intellectual_biorhythm = calculate_biorhythm(days_alive, 33)

    # Wyświetlenie wyników biorytmów
    print(f"Cześć {name}!")
    print(f"Dzisiaj jesteś na {days_alive}-ym dniu życia.")
    display_message("Fizyczny biorytm", physical_biorhythm)
    display_message("Emocjonalny biorytm", emotional_biorhythm)
    display_message("Intelektualny biorytm", intellectual_biorhythm)

    # Sprawdzenie czy kolejny dzień będzie lepszy czy gorszy
    next_physical_biorhythm = calculate_biorhythm(days_alive + 1, 23)
    if next_physical_biorhythm > physical_biorhythm:
        print("Twój fizyczny biorytm jutro będzie wyższy niż dzisiaj.")
    elif next_physical_biorhythm < physical_biorhythm:
        print("Twój fizyczny biorytm jutro będzie niższy niż dzisiaj.")
    else:
        print("Twój fizyczny biorytm jutro będzie taki sam jak dzisiaj.")

if __name__ == "__main__":
    main()

    
"""
PROMPT
Do nowej konwersacji w gpt3.5 wkleiłem polecenie z pliku z zadaniem, bez żadnych zmian, kod zadziałał za
pierwszym razem, tj bez żadnych poprawek czy dopytywania chatu

UWAGI PO PORÓWNANIU PROGRAMÓW
Wszystkie 3 programy dają takie same wyniki, AI pisząc od podstaw stworzyło przyjemniejszy do czytania
interface tekstowy oraz zaokrągliło wyniki do 2 miejsc po przecinku.

3ci kod jest napisany funkcyjnie, co może być znacznie lepsze w przypadku pracy nad tak krótkim pod względem
ilości kodu problemem.
"""