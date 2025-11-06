import requests, certifi, re
from bs4 import BeautifulSoup
import pandas as pd

# ===================== SCRAPE RECENT ===================== #
def scrape_recent_meetings():
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    html = requests.get(url, verify=certifi.where()).text
    soup = BeautifulSoup(html, "html.parser")

    meetings = []
    for sec in soup.find_all("div", class_="fomc-meeting"):
        month_tag = sec.find("div", class_="fomc-meeting__month")
        date_tag = sec.find("div", class_="fomc-meeting__date")
        if not (month_tag and date_tag):
            continue

        month = month_tag.get_text(strip=True)
        days = date_tag.get_text(strip=True)
        year = None

        for a in sec.find_all("a", href=True):
            m = re.search(r"20\d{2}", a["href"])
            if m:
                year = m.group(0)
                break
        if not year:
            continue

        meetings.append(f"{month} {days}, {year}")

    return pd.DataFrame(meetings, columns=["Meeting Date"])


# ===================== SCRAPE HISTORICAL ===================== #
def scrape_historical_meetings(start, end):
    base = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{}.htm"
    all_meetings = []

    for y in range(start, end + 1):
        url = base.format(y)
        resp = requests.get(url, verify=certifi.where())
        if resp.status_code != 200:
            print(f"Erreur {resp.status_code} pour {url}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        panels = soup.find_all("div", class_="panel-heading")
        for p in panels:
            h5 = p.find("h5")
            if not h5:
                continue
            text = h5.get_text(strip=True)

            # si pas d’année explicite dans le texte, on ajoute celle de la page
            if not re.search(r"\d{4}", text):
                text = f"{text}, {y}"

            # nettoyage léger
            text = re.sub(r"\s+", " ", text)
            all_meetings.append(text)

    return pd.DataFrame(all_meetings, columns=["Meeting Date"])


# ===================== BUILD FINAL DATASET ===================== #
def get_FOMC_meeting_dates(start_year, end_year=2019):
    df_recent = scrape_recent_meetings()
    df_hist = scrape_historical_meetings(start_year, end_year)

    df_all = pd.concat([df_hist, df_recent], ignore_index=True)
    df_all["Meeting Date"] = df_all["Meeting Date"].str.strip()
    df_all.drop_duplicates(inplace=True)
    df_all.sort_values(by="Meeting Date", inplace=True, ignore_index=True)
    return df_all


# ===================== EXECUTION ===================== #
df = get_FOMC_meeting_dates(1977, 2019)
print(df.to_csv("./test.csv"))
print(f"\nTotal meetings scraped: {len(df)}")
