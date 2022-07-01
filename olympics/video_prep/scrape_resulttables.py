import requests
from bs4 import BeautifulSoup
import pandas as pd

# Create URL object
url = 'https://skatingscores.com/q/event/?show_ranks=on&underline=&season_codes=2022&division_codes=sr&division_codes' \
      '=jr&event_codes=oly&discipline_codes=pairs&unit_country_codes=all&unit_name=%25&sort=score&limit=50&submit' \
      '=Submit '

def get_table(url):
    # Create object page
    page = requests.get(url)

    # parser-lxml = Change html to Python friendly format
    # Obtain page's information
    soup = BeautifulSoup(page.text, "html.parser")

    # Obtain information from tag <table>
    table1 = soup.find("table", {"class": "qtab"})

    # Obtain every title of columns with tag <th>
    headers = []
    for i in table1.find_all("th"):
        title = i.text
        headers.append(title)

    # Create a dataframe
    competition_results = pd.DataFrame(columns=headers)

    # Create a for loop to fill competition_results
    for j in table1.find_all("tr")[1:]:
        row_data = j.find_all("td")
        row = [i.text for i in row_data]
        length = len(competition_results)
        competition_results.loc[length] = row

    competition_results.drop('', inplace=True, axis=1)

    return competition_results
