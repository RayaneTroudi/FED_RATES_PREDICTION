# FED PREDICTION RATES

## 1. EXTRACT THE DATA 

    - We have all th interest since 1950's with daily frequency but the rates is update at each FOMC meeting. 
       But most of the macro-economique data are updated with a monthly frequence. 
       We must agregate the fed's rates monthly with a strict respect to the date of the meeting to not lost the timeline of the data.
       Hence, we scrapped the data on the FOMC web site to build a dataframe that contains all meeting date. 
       After that , we can agregate the data monthly and align all the macro-economique data monthly too. 

## 2. ALIGN MACRO ECONOMIC DATA WITH FOMC MEETING DATES

    - The main purpose is to avoid mix up of futures data that lead to false our models : 
        Example : a fomc meeting were planed to the 2000-01-01. You need to import data before this date you cannot use unployement rate at the 2000-01-01 but the month before so 1999-12-01. 