import requests
import pandas as pd
from bs4 import BeautifulSoup

page = requests.get("https://forecast.weather.gov/MapClick.php?x=276&y=148&site=lox&zmx=&zmy=&map_x=276&map_y=148#.Ya86o9DP1Pa")
print(page)

soup = BeautifulSoup(page.content, "html.parser")


div = soup.find_all("div", id="seven-day-forecast-container")

temp_high = soup.find_all('p', {'class': 'temp temp-high'})
temp_low = soup.find_all('p', {'class': 'temp temp-low'})
sent = soup.find_all('div', {'class': 'col-sm-10 forecast-text'})
days = soup.find_all('p', {'period-name'})

week = []
para_list = []
temp_high_list = []
temp_low_list = []

for x in range(9):
    week.append(days[x].text)
    para_list.append(sent[x].text)

for x in range(5):
    temp_high_list.append(temp_high[x].text)

for x in range(4):
    temp_low_list.append(temp_low[x].text)


print(week)
print(para_list)
print(temp_high_list)
print(temp_low_list)



