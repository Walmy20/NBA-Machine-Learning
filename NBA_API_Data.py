import os
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import time


SEASONS = list(range(2023, 2024))

DATA_DIR = 'data'
STANDINGS_DIR = os.path.join(DATA_DIR, 'standings')
SCORES_DIR = os.path.join(DATA_DIR, 'scores')

# function to get html

def get_html(url, selector, sleep = 7, retries = 3):
    html = None
    for i in range(1, retries+1):
        time.sleep(sleep *i)

        try:
            with sync_playwright() as p:
                browser = p.firefox.launch()
                page =  browser.new_page()
                page.goto(url)
                print(page.title())
                html = page.inner_html(selector)
        except PlaywrightTimeout:
            print(f"Timeout error on {url}")
            continue
        else:
            break
    return html


# function for scraping the  data

def scrape_season(season):
    # url
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = get_html(url, "#content .filter")

    # parsing the html
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all("a")
    href = [l["href"] for l in links]

    # creating the proper links after getting the parsing of the html
    standings_pages = [f"http://www.basketball-reference.com{l}" for l in href]
    for url in standings_pages:
        # save the file with month and year
        save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        # get box score link with values
        html = get_html(url, "#all_schedule")
        with open(save_path, "w+") as f:
            f.write(html)

# Do it for all seasons

for season in SEASONS:
    scrape_season(season)

standings_files = os.listdir(STANDINGS_DIR)

# scraping the box score by season by game

def scrape_game(standing_file):

    with open(standing_file, 'r') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all("a")
    # filter the box score links
    hrefs = [l.get("href") for l in links]
    box_scores = [l for l in hrefs if l and "boxscore" in l and ".html" in l]
    box_scores = [f"https://www.basketball-reference.com{l}" for l in box_scores]
    # saving box score using playwright

    for url in box_scores:
        save_path = os.path.join(SCORES_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        html = get_html(url,"#content")
        if not html:
            continue
        with open(save_path, "w+", encoding='utf-8') as f:
            f.write(html)

# finally get the data 

# removing any unwanted non .html
for season in SEASONS:
    files = [s for s in standings_files if str(season) in s]
    
    for f in files:
        filepath = os.path.join(STANDINGS_DIR, f)
        
        scrape_game(filepath)
