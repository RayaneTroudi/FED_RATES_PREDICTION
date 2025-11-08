import requests, certifi, re
from bs4 import BeautifulSoup
import pandas as pd

# ===================== 1936–2009 ===================== #
def scrape_historical_meetings(start=1936, end=2009):
    base = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{}.htm"
    all_meetings = []

    for y in range(start, end + 1):
        url = base.format(y)
        resp = requests.get(url, verify=certifi.where())
        if resp.status_code != 200:
            print(f"[WARN] {resp.status_code} for {url}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        panels = soup.find_all("div", class_="panel-heading")
        for p in panels:
            h5 = p.find("h5")
            if not h5:
                continue
            text = h5.get_text(strip=True)
            if not re.search(r"\d{4}", text):
                text = f"{text}, {y}"
            text = re.sub(r"\s+", " ", text)
            all_meetings.append(text)

    print(f"[INFO] Historical meetings scraped: {len(all_meetings)}")
    return pd.DataFrame(all_meetings, columns=["Meeting_Date"])


# ===================== 2010–2019 ===================== #
def scrape_mid_period_meetings():
    base = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{}.htm"
    meetings = []

    for year in range(2010, 2020):
        url = base.format(year)
        resp = requests.get(url, verify=certifi.where())
        if resp.status_code != 200:
            print(f"[WARN] {url} -> {resp.status_code}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        for h5 in soup.find_all("h5"):
            text = re.sub(r"\s+", " ", h5.get_text(strip=True))
            if not re.search(r"\d{4}", text):
                text = f"{text}, {year}"
            meetings.append(text)

    print(f"[INFO] Mid-period meetings scraped (2010–2019): {len(meetings)}")
    return pd.DataFrame(meetings, columns=["Meeting_Date"])


# ===================== 2020–2025 ===================== #
def scrape_recent_meetings():
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    resp = requests.get(url, verify=certifi.where())
    if resp.status_code != 200:
        print(f"[WARN] {resp.status_code} for {url}")
        return pd.DataFrame(columns=["Meeting_Date"])

    soup = BeautifulSoup(resp.text, "html.parser")
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

    print(f"[INFO] Recent meetings scraped: {len(meetings)}")
    return pd.DataFrame(meetings, columns=["Meeting_Date"])


# ===================== COMBINE ===================== #
def get_FOMC_meeting_dates():
    df_hist = scrape_historical_meetings(1936, 2009)
    df_mid = scrape_mid_period_meetings()
    df_recent = scrape_recent_meetings()

    df_all = pd.concat([df_hist, df_mid, df_recent], ignore_index=True)
    df_all["Meeting_Date"] = df_all["Meeting_Date"].str.strip()
    df_all.drop_duplicates(inplace=True)
    df_all.sort_values(by="Meeting_Date", inplace=True, ignore_index=True)
    print(f"[INFO] Total meetings scraped: {len(df_all)}")
    return df_all


# ===================== MAIN ===================== #
if __name__ == "__main__":
    df = get_FOMC_meeting_dates()
    df.to_csv("./data/raw/FOMC_meetings_full.csv", index=False)
