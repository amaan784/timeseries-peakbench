# Method

[README](../README.md) · [Results](results.md) · [Live report](https://amaan784.github.io/timeseries-peakbench/)

- [The question](#the-question)
- [Setup](#setup)
- [Running it](#running-it)
- [Use your own meter data](#use-your-own-meter-data)
- [Foundation models](#foundation-models)
- [Tests](#tests)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)

## The question

C&I customers in Massachusetts pay a demand charge on the single highest 15-minute interval of each month, often a third of the bill. A battery can shave that peak, but only if it discharges at the right moments, and it decides when from a day-ahead forecast. Forecasts are wrong in two ways that matter here: **magnitude** (the peak is higher or lower than predicted) and **timing** (the peak comes earlier or later). This repo measures what each costs, under different ways of turning a forecast into a dispatch decision.

## Setup

1. **Building:** a synthetic Boston-area office, 15-minute kW for 2025 plus a warm-up, with a weekday occupancy ramp, light weekends and holidays, a summer cooling season that sets the annual peak, day-level weather noise, and short random equipment spikes on weekdays (`src/data.py`). Then the same experiment on six real meters (`src/uci.py`, `src/real_meters.py`).
2. **Tariff:** illustrative Massachusetts-style time-of-use rate with a monthly demand charge on the highest 15-minute interval (`src/tariff.py`).
3. **Battery:** a 2-hour C&I battery with round-trip losses split evenly across charge and discharge, state-of-charge limits, off-peak-only charging that never raises net load above the day's target (`src/dispatch.py`).
4. **Forecasters:** seasonal naive, 28-day profile, ridge (calendar features + 1/2/7-day lags), TimesFM 3.0 and 2.5 and Mistral's Timestral 209M zero-shot on CPU, TimeGPT zero-shot via API, and a perfect-foresight oracle as the ceiling. All are day-ahead: each sees only data up to the midnight before (`src/forecast.py`, foundation models in `src/foundation.py`).
5. **Policies:** all three get the same forecast and the same daily kW target. **Window** discharges on the forecast's schedule and ignores the meter. **Threshold** discharges whenever the meter crosses the target. **Threshold + ratchet** does the same but never spends energy shaving below a peak already set this month. With a perfect forecast, window and threshold are identical, so any gap between them comes from how each handles error.

The real meters come from UCI's ElectricityLoadDiagrams20112014 (Portugal, CC BY 4.0), picked by a rule fixed before any results were run; [`data/uci/`](../data/uci/README.md) has the rule and the attribution. They use Portuguese holidays and a battery sized at 30% of each building's 2013 peak.

## Running it

```bash
make install     # core deps; no torch needed, foundation-model forecasts come from out/cache
make run         # evaluate everything -> out/results.json + tidy CSVs (~15 s on a laptop CPU)
make real        # the same on the six real meters -> out/real/ (~30 s, foundation models from cache)
make test        # pytest
make report      # out/report.html, docs/index.html, and the generated results in README.md and docs/results.md
make app         # Streamlit dashboard
```

On Windows without `make`, run the underlying commands: `pip install -r requirements.txt`, `python -m src.evaluate`, `python -m src.real_meters`, `python -m pytest -q`, `python -m src.report`, `streamlit run app.py`.

Outputs in `out/`: `results.json` (everything except per-day rows), plus tidy CSVs `summary`, `timing_sweep`, `daily`, `error_slopes`, `monthly`, `forecast_accuracy`, `battery_sensitivity`. Each has a `building` column. Nothing downstream (dashboard, report, README) types in a number; all of it is read from these files or recomputed by the same functions.

## Use your own meter data

Change one line in [`src/config.py`](../src/config.py):

```python
DATA_SOURCE = "data/my_building_15min.csv"
```

`load_interval_csv()` reads a utility interval export or an ISO-NE load file into the same shape as the synthetic data: a 15-minute DatetimeIndex and one `kw` column. It auto-detects the timestamp and value columns, converts kWh or MW(h) to kW, maps any interval length onto 15 minutes, converts UTC offsets to local time (default `America/New_York`), and handles the DST hours. The evaluation uses the last 365 complete days that have 35 days of history before them. The no-spike counterfactual only exists for synthetic data.

## Foundation models

**TimesFM** (`TimesFMZeroShot` / `timesfm_zeroshot()` in `src/foundation.py`) runs Google's checkpoint locally on CPU through the current `timesfm` package (3.0.x): `TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cpu")`, then `predict_batch(contexts, horizon=96)` with a 28-day trailing context, batched 32 days at a time. Version 2.5 uses `TimesFM_2p5_200M_torch` with a compiled `ForecastConfig`. A year of day-ahead forecasts took about five minutes per model on an 8-core laptop CPU when the cache was built.

**Timestral** (`TimestralZeroShot` / `timestral_zeroshot()`) runs Mistral's 209M checkpoint locally on CPU through the `timestral` package (from GitHub; not on PyPI yet, so `requirements-fm.txt` pins the commit): `TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cpu")`, then `forecast(contexts, prediction_length=96)` on the same 28-day contexts, 32 days per batch. It returns 21 quantiles; the median is the forecast and the 10th and 90th percentiles go in the cache. The 5M, 28M and 123M sizes work too (`TimestralZeroShot("28M")`). A year took about three minutes per dataset on the same CPU.

```bash
make install-fm   # pip install "timesfm[torch]", timestral (from GitHub), nixtla
```

**TimeGPT** (`TimeGPTZeroShot` / `timegpt_zeroshot()`) uses Nixtla's client (`nixtla>=0.9`). It reads `NIXTLA_API_KEY` from the environment or a gitignored `.env` file and sends 32 days per request as separate series. The model is `timegpt-2.1` unless `TIMEGPT_MODEL` says otherwise. With no key it logs a message and serves only what is cached.

**Cache:** one Parquet file per model and dataset, `out/cache/<model>/<data fingerprint>.parquet`, with a row per 15-minute step holding the median forecast plus 10th/90th percentiles; forecasts are looked up by date inside it. The fingerprint is a hash of the load data, so a cache can never be reused against different data. It is committed, so the evaluation, dashboard and report all work without the checkpoint or a key. To add TimeGPT, set the key and run `make forecasts && make report`.

## Tests

`tests/` covers:

- state of charge never leaves its bounds
- energy balance: what goes in, minus losses each way, equals what comes out plus whatever is left in the battery, and discharged energy is roughly charged energy times the round-trip efficiency
- charging never pushes net load above the day's target, and only happens off-peak
- power limit and no export to the grid
- window and threshold give identical results with a perfect forecast
- **no forecaster can see data at or after its target day**: the future gets overwritten with garbage and every forecast has to come out unchanged, including the batched foundation-model path
- the forecast cache key, TimeGPT skipping cleanly without a key, and the Timestral wrapper (output shapes, NaN padding, skipping without the package) against a fake model
- `bill()`: the demand charge is the month's highest interval times the rate
- the generated results only ever replace the text between their two marker comments

## Repository layout

```
src/
  config.py      data source switch, paths
  data.py        synthetic building + load_interval_csv()
  forecast.py    forecasters (oracle, seasonal naive, profile, ridge) and the day-ahead runner
  foundation.py  TimesFM, Timestral and TimeGPT, zero-shot, with the forecast cache
  tariff.py      Tariff dataclass and bill()
  dispatch.py    Battery, daily target bisection, simulate()
  evaluate.py    forecaster x policy x timing grid -> out/
  charts.py      plotly figures shared by the dashboard and report
  report.py      out/report.html, docs/index.html, the generated results in README.md and docs/results.md
  uci.py         download + clean the UCI meters, pick buildings by a fixed rule
  real_meters.py the same experiment on the real meters -> out/real/
app.py           Streamlit dashboard
data/uci/        six cleaned real meters + how they were picked
docs/            the report (GitHub Pages), results.md, method.md
tests/           pytest
out/             results, CSVs, report, forecast cache (committed)
```

## Limitations

- **Mostly synthetic data.** The main experiment is one generated building and one random seed, with a spike process I chose, which is why the spike-free twin is reported alongside. The six real meters are a check: anonymous clients, one country, one year, and an MA tariff applied to Portuguese buildings. MT_151 has a day of zero readings and MT_164 a few; they were left in, as real data. No confidence intervals: forecaster differences of a few percentage points are within noise.
- **Illustrative tariff.** No customer charges, demand ratchets, coincident-peak or capacity charges, and no real utility's filed rates.
- **No degradation model.** Cycling is free, so the policies that cycle more are not penalised for it.
- **No grid-services revenue.** No demand response, frequency regulation or wholesale arbitrage stacked on bill savings.
- **Single meter, single battery,** no portfolio effects.
- **Day-ahead only.** No intraday updates, which any production controller would add.
- **TimeGPT** is implemented and tested but only runs with a `NIXTLA_API_KEY`. Without one it drops out of the comparison rather than being estimated (the [results](results.md) say whether it ran).
- **TimesFM 3.0 weights** are licensed for non-commercial use only. TimesFM 2.5 weights are Apache-2.0, which is why both are here. **Timestral** weights are under Mistral's modified MIT licence (see the model card).
