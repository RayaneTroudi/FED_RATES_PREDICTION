# ---
# CLEAN AND STRUCTURE THE RAW FOMC MEETING DATES
# ---
import re
import pandas as pd
from datetime import datetime
from src.preprocessing.scraper import get_FOMC_meeting_dates

# --- MONTH STRING TO INTEGER ---
def month_str_to_int(mois):
    mapping = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12
    }
    return mapping.get(mois.lower(), float('nan'))


# --- CLEAN & CONVERT MEETING DATES ---
def build_fed_meeting_dates():
    df_fomc_meeting = get_FOMC_meeting_dates(start_year=1977, end_year=2019)
    cleaned_dates = []

    for raw in df_fomc_meeting["Meeting Date"]:
        try:
            # ignore conference calls or notation votes (non-FOMC meetings)
            if any(x in raw.lower() for x in ["conference", "notation", "call", "unscheduled"]):
                continue

            # isolate year
            m_year = re.search(r"(19|20)\d{2}", raw)
            if not m_year:
                continue
            year = int(m_year.group(0))

            # isolate month
            m_month = re.search(r"([A-Za-z]{3,9})", raw)
            if not m_month:
                continue
            month = month_str_to_int(m_month.group(1))

            # isolate day range (ex: 30–31 -> 31)
            m_day = re.search(r"(\d{1,2})(?:[-–/]\d{1,2})?", raw)
            if not m_day:
                continue
            # dernière valeur si range (31 dans 30–31)
            day_part = re.split(r"[-–/]", m_day.group(0))[-1]
            day = int(day_part)

            date_str = f"{year}-{month:02d}-{day:02d}"
            cleaned_dates.append(date_str)

        except Exception:
            continue

    df_clean = pd.DataFrame(sorted(set(cleaned_dates)), columns=["Meeting_Date"])
    df_clean["Meeting_Date"] = pd.to_datetime(df_clean["Meeting_Date"])
    df_clean.sort_values(by="Meeting_Date", inplace=True, ignore_index=True)

    output_path = "/Users/rayane_macbook_pro/Documents/Prog_ENSAE/ML_FOR_PORTF_TRADING/FED_PROJECT/data/processed/FOMC_MEETING.csv"
    df_clean.to_csv(output_path, index=False)

    return df_clean


# --- BUILD MERGED FED FUNDS RATES AT MEETING DATES ---
def build_rates_at_meeting_dates():
    path_meetings = "/Users/rayane_macbook_pro/Documents/Prog_ENSAE/ML_FOR_PORTF_TRADING/FED_PROJECT/data/processed/FOMC_MEETING.csv"
    path_rates = "/Users/rayane_macbook_pro/Documents/Prog_ENSAE/ML_FOR_PORTF_TRADING/FED_PROJECT/data/raw/DFF.csv"
    output_path = "/Users/rayane_macbook_pro/Documents/Prog_ENSAE/ML_FOR_PORTF_TRADING/FED_PROJECT/data/processed/DFF_PROCESSED.csv"

    # load datasets
    df_meet = pd.read_csv(path_meetings)
    df_rates = pd.read_csv(path_rates)

    # harmonize column names
    df_meet.rename(columns={"Meeting_Date": "DATE"}, inplace=True)
    df_rates.rename(columns={"observation_date": "DATE"}, inplace=True)

    df_meet["DATE"] = pd.to_datetime(df_meet["DATE"])
    df_rates["DATE"] = pd.to_datetime(df_rates["DATE"])

    # sort before merge_asof
    df_meet.sort_values("DATE", inplace=True)
    df_rates.sort_values("DATE", inplace=True)

    # merge: find first daily rate *after* each meeting
    df_merged = pd.merge_asof(
        df_meet,
        df_rates,
        on="DATE",
        direction="forward",
        allow_exact_matches=False
    )

    df_merged.rename(columns={"DATE": "observation_date"}, inplace=True)
    df_merged.to_csv(output_path, index=False)
    return df_merged


# === EXECUTION ===
if __name__ == "__main__":
    df_clean = build_fed_meeting_dates()
    print(f"Meetings processed: {len(df_clean)}")
    df_final = build_rates_at_meeting_dates()
    print(f"Rates merged: {len(df_final)}")
