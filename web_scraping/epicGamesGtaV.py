import requests
from bs4 import BeautifulSoup
from datetime import datetime
import webbrowser
import time

url = "https://www.epicgames.com/store/en-US/free-games"

while True:
    try:
        print(datetime.now(), end=" - ")
        page = requests.get(url)
        print(page.status_code)
        if page.status_code == 200:
            webbrowser.open(url, new=2)
        time.sleep(5)
    except Exception as e:
        print(e)

# soup = BeautifulSoup(page.content, "html.parser")

# print(soup.prettify())
