# ---
# THIS FILE IS USED TO CLEAN THE RAW DATA AND CREATE A CLEAN VERSION OF THE DATASET
# --- 
from preprocessing.scraper import get_FOMC_meeting_dates
import pandas as pd 
# --- FUNCTION TO BUILD THE FED RATES MONTHLY

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

def build_fed_meeting_dates():
    
    df_fomc_meeting = get_FOMC_meeting_dates(start_date=2000, end_date=2019)
    df_fomc_meeting_preprocessed = pd.DataFrame()
    arr_fomc_meeting_date = []

    for meeting_date in df_fomc_meeting["Meeting Date"]:
        try:
            year = meeting_date.split(', ')[1]
            day = meeting_date.split(', ')[0].split(' ')[1].split('-')
            if len(day) == 1:
                day_ = day[0]
            else:
                day_ = day[1][0:2]
                print(day_)
            month_str = meeting_date.split(', ')[0].split(' ')[0].split('/')[0]
            month_num = month_str_to_int(month_str)
            arr_fomc_meeting_date.append(f"{year}-{int(month_num):02d}-{int(day_):02d}")
        except Exception as e:
            print(f"Erreur sur {meeting_date}: {e}")
            continue

    df_fomc_meeting_preprocessed["Meeting_Date"] = arr_fomc_meeting_date
    df_fomc_meeting_preprocessed.sort_values(by='Meeting_Date', inplace=True)
    df_fomc_meeting_preprocessed.to_csv('/Users/rayane_macbook_pro/Documents/Prog_ENSAE/ML_FOR_PORTF_TRADING/FED_PROJECT/data/processed/FOMC_MEETING.csv', index=False)
    return df_fomc_meeting_preprocessed



def build_rates_at_meeting_dates():

    # Load meeting dates
    df_meeting_dates_rates = pd.read_csv(
        "/Users/rayane_macbook_pro/Documents/Prog_ENSAE/ML_FOR_PORTF_TRADING/FED_PROJECT/data/processed/FOMC_MEETING.csv",
        header=0
    )

    # Load daily rates
    df_fomc_rates_daily = pd.read_csv(
        "/Users/rayane_macbook_pro/Documents/Prog_ENSAE/ML_FOR_PORTF_TRADING/FED_PROJECT/data/raw/DFF.csv",
        header=0
    )

    # Rename and convert to prepare the merge
    df_meeting_dates_rates.rename(columns={"Meeting_Date": "DATE"}, inplace=True)
    df_fomc_rates_daily.rename(columns={"observation_date": "DATE"}, inplace=True)
    df_meeting_dates_rates["DATE"] = pd.to_datetime(df_meeting_dates_rates["DATE"])
    df_fomc_rates_daily["DATE"] = pd.to_datetime(df_fomc_rates_daily["DATE"])

    # Sort before merge_asof
    df_meeting_dates_rates.sort_values("DATE", inplace=True)
    df_fomc_rates_daily.sort_values("DATE", inplace=True)

    # Merge to get rate *after* meeting date
    df_merged = pd.merge_asof(
        df_meeting_dates_rates,
        df_fomc_rates_daily,
        on="DATE",
        direction="forward",
        allow_exact_matches=False  # force strict "date after"
    )

    # Export
    df_merged.rename(columns={"DATE": "observation_date"}, inplace=True)
    df_merged.to_csv(
        "/Users/rayane_macbook_pro/Documents/Prog_ENSAE/ML_FOR_PORTF_TRADING/FED_PROJECT/data/processed/DFF_PROCESSED.csv",
        index=False
    )

    return df_merged

build_rates_at_meeting_dates()

