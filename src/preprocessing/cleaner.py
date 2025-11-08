import re
import pandas as pd
from datetime import datetime

# =============================================================================
# 1. EXTRACTION ET NETTOYAGE DES DATES DE MEETING FOMC
# =============================================================================

df_raw = pd.read_csv("./data/raw/FOMC_meetings_full.csv", header=None, names=["Meeting_Date"])

pattern_meeting = re.compile(r"^([A-Za-z]+)\s+(\d{1,2})(?:-(\d{1,2}))?\s+Meeting\s*-\s*(\d{4})$")
pattern_range = re.compile(r'"?([A-Za-z/]+)\s+(\d{1,2})(?:-(\d{1,2})\*?)?,\s*(\d{4})"?')

months = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
    "Jan/Feb": (1, 2), "Feb/Mar": (2, 3), "Mar/Apr": (3, 4), "Apr/May": (4, 5),
    "May/Jun": (5, 6), "Jun/Jul": (6, 7), "Jul/Aug": (7, 8), "Aug/Sep": (8, 9),
    "Sep/Oct": (9, 10), "Oct/Nov": (10, 11), "Nov/Dec": (11, 12)
}

dates = []

for val in df_raw["Meeting_Date"]:
    if pd.isna(val):
        continue
    val = str(val).strip()
    if "Conference Call" in val:
        continue

    # Cas 1 : "February 1-2 Meeting - 2000"
    m = pattern_meeting.match(val)
    if m:
        month, day1, day2, year = m.groups()
        if month in months:
            try:
                day = int(day2) if day2 else int(day1)
                dt = datetime(int(year), months[month], day)
            except ValueError:
                dt = datetime(int(year), months[month], 1)
            dates.append(dt)
        continue

    # Cas 2 : "Apr/May 30-1, 2024"
    r = pattern_range.search(val)
    if r:
        month, day1, day2, year = r.groups()
        if month in months:
            try:
                month_val = months[month]
                if isinstance(month_val, tuple):
                    month_val = month_val[1]  # mois le plus tardif
                day = int(day2) if day2 else int(day1)
                dt = datetime(int(year), month_val, day)
            except ValueError:
                dt = datetime(int(year), month_val, 1)
            dates.append(dt)

df_clean = pd.DataFrame(sorted(dates), columns=["observation_date"])
df_clean["observation_date"] = pd.to_datetime(df_clean["observation_date"])
df_clean.to_csv("./data/processed/FOMC_MEETINGS.csv", index=False)

print(f"{len(df_clean)} dates extraites et sauvegardées dans ./data/processed/FOMC_MEETINGS.csv")
print(df_clean.tail())

# =============================================================================
# 2. MERGE-ASOF AVEC LES TAUX DFF (JOUR SUIVANT)
# =============================================================================

meetings = pd.read_csv("./data/processed/FOMC_MEETINGS.csv")
dff = pd.read_csv("./data/raw/DFF.csv")

meetings["observation_date"] = pd.to_datetime(meetings["observation_date"])
dff["observation_date"] = pd.to_datetime(dff["observation_date"])
dff = dff.dropna(subset=["observation_date", "DFF"])

meetings = meetings.sort_values("observation_date")
dff = dff.sort_values("observation_date")

meetings["next_day"] = meetings["observation_date"] + pd.Timedelta(days=1)

merged = pd.merge_asof(
    meetings,
    dff,
    left_on="next_day",
    right_on="observation_date",
    direction="forward",
    tolerance=pd.Timedelta("7D")
)

merged = merged[["observation_date_x", "DFF"]].rename(
    columns={"observation_date_x": "meeting_date", "DFF": "next_day_rate"}
)

# =============================================================================
# 3. FILTRE SUR UNE PLAGE DE DATES
# =============================================================================

# borne de filtrage (à modifier selon besoin)
start_date = "1990-01-01"
end_date = "2025-12-31"

merged = merged[
    (merged["meeting_date"] >= start_date) &
    (merged["meeting_date"] <= end_date)
]

# =============================================================================
# 4. EXPORT FINAL
# =============================================================================
merged.rename(columns={"meeting_date":"observation_date","next_day_rate":"DFF"},inplace=True)
merged.to_csv("./data/processed/DFF_PROCESSED.csv", index=False)

print(f"{len(merged)} lignes dans la plage [{start_date}, {end_date}] sauvegardées dans ./data/processed/DFF_PROCESSED.csv")
print(merged.head(10))
