import re
from typing import List, Optional
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class FomcUtils:
    """Utilities for processing FOMC data."""

    regexp = re.compile(r"\s+", re.UNICODE)

    @staticmethod
    def get_fomc_urls(from_year: int = 1999, switch_year: Optional[int] = None) -> List[str]:
        """
        Collect URLs of FOMC statements from both the modern HTML calendar and legacy archives.
        Covers pre-2005 (fomc/*.htm, boarddocs/general) and recent press releases.
        """
        if switch_year is None:
            from datetime import datetime
            switch_year = datetime.now().year - 5

        urls_ = []

        # --- Modern (2010+)
        calendar_url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        r = requests.get(calendar_url)
        soup = BeautifulSoup(r.text, "html.parser")
        contents = soup.find_all("a", href=re.compile(r"^/newsevents/pressreleases/monetary\d{8}[ax]\.htm"))
        urls_.extend(c.attrs["href"] for c in contents)

        # --- Historical (1999–2018)
        for year in range(from_year, switch_year):
            yearly_url = f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
            try:
                r_year = requests.get(yearly_url, timeout=10)
                if r_year.status_code == 200:
                    soup_yearly = BeautifulSoup(r_year.text, "html.parser")
                    yearly_links = soup_yearly.find_all("a", string="Statement")
                    urls_.extend(l.attrs["href"] for l in yearly_links)
            except Exception:
                continue

        # --- Legacy fallback (1990–2005)
        urls_.extend(FomcUtils.get_legacy_urls())

        # Normalize and deduplicate
        urls = ["https://www.federalreserve.gov" + u if u.startswith("/") else u for u in urls_]
        urls = list(dict.fromkeys(urls))
        return urls

    @staticmethod
    def get_legacy_urls() -> List[str]:
        """Hardcoded legacy FOMC statement URLs (1994–2005)."""
        # (même liste que précédemment, conservée intégralement)
        return [
            "https://www.federalreserve.gov/fomc/19940204default.htm",
            "https://www.federalreserve.gov/fomc/19940322default.htm",
            "https://www.federalreserve.gov/fomc/19940418default.htm",
            "https://www.federalreserve.gov/fomc/19940517default.htm",
            "https://www.federalreserve.gov/fomc/19940816default.htm",
            "https://www.federalreserve.gov/fomc/19941115default.htm",
            "https://www.federalreserve.gov/fomc/19950201default.htm",
            "https://www.federalreserve.gov/fomc/19950706default.htm",
            "https://www.federalreserve.gov/fomc/19951219default.htm",
            "https://www.federalreserve.gov/fomc/19960131DEFAULT.htm",
            "https://www.federalreserve.gov/boarddocs/press/general/1997/19970325/",
            "https://www.federalreserve.gov/boarddocs/press/general/1998/19980929/",
            "https://www.federalreserve.gov/boarddocs/press/general/1998/19981015/",
            "https://www.federalreserve.gov/boarddocs/press/general/1998/19981117/",
            "https://www.federalreserve.gov/boarddocs/press/general/1999/19990518/",
            "https://www.federalreserve.gov/boarddocs/press/general/1999/19990630/",
            "https://www.federalreserve.gov/boarddocs/press/general/1999/19990824/",
            "https://www.federalreserve.gov/boarddocs/press/general/1999/19991005/",
            "https://www.federalreserve.gov/boarddocs/press/general/1999/19991116/",
            "https://www.federalreserve.gov/boarddocs/press/general/1999/19991221/",
            "https://www.federalreserve.gov/boarddocs/press/general/2000/20000202/",
            "https://www.federalreserve.gov/boarddocs/press/general/2000/20000321/",
            "https://www.federalreserve.gov/boarddocs/press/general/2000/20000516/",
            "https://www.federalreserve.gov/boarddocs/press/general/2000/20000628/",
            "https://www.federalreserve.gov/boarddocs/press/general/2000/20000822/",
            "https://www.federalreserve.gov/boarddocs/press/general/2000/20001003/",
            "https://www.federalreserve.gov/boarddocs/press/general/2000/20001115/",
            "https://www.federalreserve.gov/boarddocs/press/general/2000/20001219/",
            "https://www.federalreserve.gov/boarddocs/press/general/2001/20010103/",
            "https://www.federalreserve.gov/boarddocs/press/general/2001/20010131/",
            "https://www.federalreserve.gov/boarddocs/press/general/2001/20010320/",
            "https://www.federalreserve.gov/boarddocs/press/general/2001/20010418/",
            "https://www.federalreserve.gov/boarddocs/press/general/2001/20010515/",
            "https://www.federalreserve.gov/boarddocs/press/general/2001/20010627/",
            "https://www.federalreserve.gov/boarddocs/press/general/2001/20010821/",
            "https://www.federalreserve.gov/boarddocs/press/general/2001/20010917/",
            "https://www.federalreserve.gov/boarddocs/press/general/2001/20011002/",
            "https://www.federalreserve.gov/boarddocs/press/general/2001/20011106/",
            "https://www.federalreserve.gov/boarddocs/press/general/2001/20011211/",
            "https://www.federalreserve.gov/boarddocs/press/general/2002/20020130/",
            "https://www.federalreserve.gov/boarddocs/press/general/2002/20020319/",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2002/20020507/",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2002/20020626/",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2002/20020813/",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2002/20020924/",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2002/20021106/",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2002/20021210/",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2003/20030129/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2003/20030318/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2003/20030506/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2003/20030625/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2003/20030812/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2003/20030916/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2003/20031028/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2003/20031209/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20040128/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20040316/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20040504/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20040630/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20040810/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20040921/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20041110/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2004/20041214/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20050202/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20050322/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20050503/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20050630/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20050809/default.htm",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20050920/",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20051101/",
            "https://www.federalreserve.gov/boarddocs/press/monetary/2005/20051213/",
        ]

    @staticmethod
    def extract_meeting_date(url: str, text: List[str]) -> Optional[str]:
        """Try to infer meeting date either from text or URL."""
        # Try pattern inside text
        joined = " ".join(text)
        pattern = re.compile(
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:–|-| and )?\d{0,2},\s+\d{4}"
        )
        match = pattern.search(joined)
        if match:
            try:
                return pd.to_datetime(match.group(0)).strftime("%Y-%m-%d")
            except Exception:
                pass

        # Fallback: infer from digits in URL
        m = re.search(r"(\d{8})", url)
        if m:
            try:
                return pd.to_datetime(m.group(1)).strftime("%Y-%m-%d")
            except Exception:
                return None
        return None

    @staticmethod
    def feature_extraction(corpus: List[List[str]], urls: List[str]) -> pd.DataFrame:
        """Extract meeting_date and text."""
        cleaned_texts, dates = [], []
        for paragraphs, url in zip(corpus, urls):
            text = " ".join(FomcUtils.regexp.sub(" ", p) for p in paragraphs if len(p) > 2)
            date = FomcUtils.extract_meeting_date(url, paragraphs)
            cleaned_texts.append(text)
            dates.append(date)
        return pd.DataFrame({"meeting_date": dates, "text": cleaned_texts, "url": urls})


def load_fomc_statements(from_year=1990, progress_bar=True):
    """Full scraping pipeline for FOMC statements."""
    urls = FomcUtils.get_fomc_urls(from_year=from_year)
    urls_iter = tqdm(urls) if progress_bar else urls

    corpus = []
    for url in urls_iter:
        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            corpus.append(paragraphs)
        except Exception:
            corpus.append([])

    df = FomcUtils.feature_extraction(corpus, urls)
    df = df.sort_values("meeting_date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_fomc_statements(from_year=1990, progress_bar=True)
    print(df.head(10))
    df.to_csv("./data/processed/FOMC_statements.csv", index=False)
    print("Fichier sauvegardé : ./data/processed/FOMC_statements.csv")
