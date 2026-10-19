# Real meter data

Six buildings from **ElectricityLoadDiagrams20112014** (Artur Trindade, 2015), UCI Machine Learning Repository,
https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014. License: CC BY 4.0.

The original has 370 anonymous clients in Portugal, 15-minute kW readings from 2011 to 2014. These files are a
cleaned subset, 2013-10-01 to 2014-12-31, one CSV per building with columns `timestamp,kw`.

What was changed from the original:

- timestamps moved from end-of-interval to start-of-interval
- spring DST: the four zero readings were treated as missing and interpolated
- autumn DST: the hour that holds two hours of readings was halved
- values rounded to 0.001 kW

How the six were picked (`src/uci.py`, fixed before any battery results were run):

- under 0.5% zero readings across 2013-2014
- 2013 peak between 100 and 2,000 kW
- business-day daytime load at least 1.3x the weekend daytime load (office-like)
- 28 of the 370 clients pass; six were drawn at random with seed 2014

`selection.json` has the rule and the stats for each building. To rebuild everything from the original file:

```bash
python -m src.uci
```

It downloads the 250 MB zip into `data/raw/` (not committed).
