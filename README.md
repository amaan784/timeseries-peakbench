# Time Series Peakbench

Benchmarking load forecasters for battery peak shaving.

**When a commercial battery uses a load forecast to cut demand charges, how much money does forecast error actually cost?**

A small, reproducible simulation study: one synthetic building for the main experiment, then six real metered buildings as a check. It answers with dollars rather than MAPE, and it tests a specific claim: that most of the value is not in a better forecast but in a dispatch policy that degrades gracefully when the forecast is wrong. The experiment was built so that claim could fail, and in part it did. The results below say where.

- **Report:** [`docs/index.html`](docs/index.html), a standalone page with every chart (serve `docs/` with GitHub Pages).
- **Dashboard:** `make app` (Streamlit). Every number is recomputed live from the sidebar's battery, tariff and policy.

> **Data:** the main study uses a generated building. The check uses real 15-minute meters from a public dataset (UCI ElectricityLoadDiagrams20112014, Portugal, CC BY 4.0, see [`data/uci/`](data/uci/README.md)). The tariff is illustrative in both. Your own interval data swaps in with one line (see [Use your own meter data](#use-your-own-meter-data)).

## The question

C&I customers in Massachusetts pay a demand charge on the single highest 15-minute interval of each month, often a third of the bill. A battery can shave that peak, but only if it discharges at the right moments, and it decides when from a day-ahead forecast. Forecasts are wrong in two ways that matter here: **magnitude** (the peak is higher or lower than predicted) and **timing** (the peak comes earlier or later). This repo measures what each costs, under different ways of turning a forecast into a dispatch decision.

## Setup in five lines

1. **Building:** a synthetic Boston-area office, 15-minute kW for 2025 plus a warm-up, with a weekday occupancy ramp, light weekends and holidays, a summer cooling season that sets the annual peak, day-level weather noise, and short random equipment spikes on weekdays (`src/data.py`). Then the same experiment on six real meters (`src/uci.py`, `src/real_meters.py`).
2. **Tariff:** illustrative Massachusetts-style time-of-use rate with a monthly demand charge on the highest 15-minute interval (`src/tariff.py`).
3. **Battery:** a 2-hour C&I battery with round-trip losses split evenly across charge and discharge, state-of-charge limits, off-peak-only charging that never raises net load above the day's target (`src/dispatch.py`).
4. **Forecasters:** seasonal naive, 28-day profile, ridge (calendar features + 1/2/7-day lags), TimesFM 3.0 and 2.5 zero-shot on CPU, TimeGPT zero-shot via API, and a perfect-foresight oracle as the ceiling. All are day-ahead: each sees only data up to the midnight before (`src/forecast.py`, foundation models in `src/foundation.py`).
5. **Policies:** all three get the same forecast and the same daily kW target. **Window** discharges on the forecast's schedule and ignores the meter. **Threshold** discharges whenever the meter crosses the target. **Threshold + ratchet** does the same but never spends energy shaving below a peak already set this month. With a perfect forecast, window and threshold are identical, so any gap between them comes from how each handles error.

## Results

<!-- results:start -->
**Choosing the dispatch policy is worth $20k a year. Choosing the forecaster is worth $2.8k.**

1. **Policy beats forecaster.** Holding the forecaster fixed (Ridge), the three dispatch policies are $20k a year apart. Holding the policy fixed (threshold + ratchet), the five forecasters are $2.8k apart. In every case the spread across policies exceeds the spread across forecasters, by 2.3x to 6.5x.
2. **The obvious threshold policy is the worst one.** Setting each day's target as low as the forecast says the battery can hold captures only 16-22% of perfect-foresight savings. It spends the whole battery on the forecast's peak, so when the real peak is higher the battery is empty before it arrives: it ran out with the meter still above target on 47-72% of business days (the ratchet: 5-7%).
3. **Reading the meter only pays if the policy respects the bill.** Adding one rule (never shave below a peak already set this month) lifts the same closed-loop policy to 62-67%, against 32-35% for a forecast-scheduled window.
4. **Timing error is what kills schedules.** A perfect forecast running 30 minutes late costs the window policy $38k a year; the meter-driven policies don't move.
5. **But no policy wins everywhere.** Remove the short equipment spikes from the same building and the ranking flips: the window schedule captures 67-77%, the ratchet 43-54%. Spikes are what a forecast can't place and a meter can catch; on a smooth load, the risk is under-forecasting the level, and a schedule bounded by its plan degrades more gently.
6. **Real meters back this up.** On metered 15-minute data from six buildings in Portugal (UCI, 2014, picked by a fixed rule), threshold + ratchet was the best policy on six of six with the Ridge forecast, capturing 47-93% of the best perfect-foresight savings. A perfect forecast 30 minutes off cost the window schedule 35-70% of its savings. Real loads are harder to forecast (Ridge MAPE 7-21% on business days). TimesFM 3.0 had the lower MAPE on six of six, but beat ridge on peak error on only two, and moved threshold + ratchet savings by -$1.5k to +$1.2k a year per building.
7. **Foundation models did not change the answer.** On the synthetic building, neither foundation model beat ridge on day-ahead peak error (71 and 66 vs 65 kW mean absolute error), despite a better average MAPE. Under threshold + ratchet they moved annual savings by +$893 (TimesFM 3.0), +$2.8k (TimesFM 2.5), against $20k between policies. TimeGPT is implemented (`timegpt_zeroshot()`) but did not run for this report: no `NIXTLA_API_KEY` was available, so it is left out of every comparison rather than estimated.

#### Annual savings by forecaster and policy (with equipment spikes, the main case)

Each cell: savings vs the no-battery bill, and the share of perfect-foresight savings captured under that policy.

| Forecaster | Threshold + ratchet | Threshold | Window | Peak error, MAE | MAPE, business days |
|---|---:|---:|---:|---:|---:|
| Oracle (perfect foresight) | $50,171 (ceiling) | $52,197 (ceiling) | $52,197 (ceiling) | 0 kW | 0% |
| Seasonal naive | $31,278 (62%) | $9,655 (18%) | $16,569 (32%) | 75 kW | 6.1% |
| 28-day profile | $33,085 (66%) | $10,689 (20%) | $18,050 (35%) | 61 kW | 5.0% |
| Ridge | $31,024 (62%) | $11,314 (22%) | $18,046 (35%) | 65 kW | 5.7% |
| TimesFM 3.0 (zero-shot) | $31,917 (64%) | $8,271 (16%) | $18,032 (35%) | 71 kW | 4.7% |
| TimesFM 2.5 (zero-shot) | $33,863 (67%) | $10,903 (21%) | $16,562 (32%) | 66 kW | 4.7% |

Bill without a battery: $579,413 a year. Battery 500 kWh / 250 kW, 88% round trip. Demand charge $22.00/kW-month. Business days a meter-driven policy ran out of energy with the meter still above its target: threshold 47-72%; threshold + ratchet 5-7%.

#### Smooth (no equipment spikes): the same building with the equipment spikes removed

| Forecaster | Threshold + ratchet | Threshold | Window |
|---|---:|---:|---:|
| Seasonal naive | $9,977 (43%) | $4,939 (20%) | $16,854 (67%) |
| 28-day profile | $11,967 (52%) | $5,313 (21%) | $19,384 (77%) |
| Ridge | $10,795 (47%) | $4,926 (20%) | $19,144 (76%) |
| TimesFM 3.0 (zero-shot) | $12,544 (54%) | $5,725 (23%) | $19,469 (77%) |
| TimesFM 2.5 (zero-shot) | $11,735 (51%) | $6,203 (25%) | $19,402 (77%) |

#### Timing error: a perfect-shape forecast shifted in time (annual savings)

| Policy | -60 min | -45 min | -30 min | -15 min | 0 | +15 min | +30 min | +45 min | +60 min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Threshold + ratchet | $50,171 | $50,171 | $50,171 | $50,171 | $50,171 | $50,171 | $50,171 | $50,171 | $50,171 |
| Threshold | $52,197 | $52,197 | $52,197 | $52,197 | $52,197 | $52,197 | $52,197 | $52,197 | $52,197 |
| Window | $12,370 | $12,452 | $13,986 | $15,027 | $52,197 | $16,648 | $14,863 | $14,122 | $13,561 |

#### Foundation models vs ridge (synthetic building)

**Better average accuracy did not buy better peaks, and the dollars moved by at most $3.0k, against $20k for the choice of policy.**

- **TimesFM 3.0 (zero-shot)** did not beat ridge on day-ahead peak error (71 vs 65 kW mean absolute error; its peaks run 66 kW low on average), while its business-day MAPE was better (4.7% vs 5.7%). Dollars vs ridge: +$893 under threshold + ratchet, -$3.0k under threshold, -$14 under window.
- **TimesFM 2.5 (zero-shot)** did not beat ridge on day-ahead peak error (66 vs 65 kW mean absolute error; its peaks run 58 kW low on average), while its business-day MAPE was better (4.7% vs 5.7%). Dollars vs ridge: +$2.8k under threshold + ratchet, -$411 under threshold, -$1.5k under window.
- TimeGPT is implemented (`timegpt_zeroshot()`) but did not run for this report: no `NIXTLA_API_KEY` was available, so it is left out of every comparison rather than estimated.

#### Battery size shifts the numbers, not the ranking. The building's load shape changes the ranking.

| Case | Threshold + ratchet | Threshold | Window |
|---|---:|---:|---:|
| Half-size battery (250 kWh / 125 kW) | 78-80% | 14-25% | 21-28% |
| Base battery (500 kWh / 250 kW) | 62-67% | 16-22% | 32-35% |
| Double-size battery (1,000 kWh / 500 kW) | 57-68% | 25-32% | 47-50% |
| Smooth (no equipment spikes), base battery | 43-54% | 20-25% | 67-77% |

**Does the policy still matter more than the forecaster?** In every case the spread across policies exceeds the spread across forecasters, by 2.3x to 6.5x. The two best policies come closer than that, and in one case (double-size battery, with equipment spikes) closer to each other than the best and worst forecasters are.

| Case | Spread across policies | Best vs second-best policy | Spread across forecasters | Ratio |
|---|---:|---:|---:|---:|
| Half-size battery, with equipment spikes | $19,203 | $16,360 | $3,643 | 5.3x |
| Base battery, with equipment spikes | $19,710 | $12,978 | $3,043 | 6.5x |
| Double-size battery, with equipment spikes | $15,227 | $3,874 | $6,761 | 2.3x |
| Base battery, smooth (no equipment spikes) | $11,916 | $6,878 | $2,615 | 4.6x |

Annual savings. Policy columns: the smallest gap over all forecasters. Forecaster column: the largest gap over all policies.

#### Real meters: six UCI buildings, Portugal, 2014

**On all six real buildings, threshold + ratchet is the best policy, and the policy spread beats the forecaster spread on six of six.**

From UCI's ElectricityLoadDiagrams20112014 (CC BY 4.0), picked by the fixed rule in `src/uci.py`. Battery sized at 30% of each building's 2013 peak, two hours of storage. Policy columns are savings as a share of the best perfect-foresight result across the three policies, with the Ridge forecast.

| Building | 2014 peak | Battery | MAPE | Threshold + ratchet | Threshold | Window | Policy spread | Forecaster spread | TimesFM 3.0 vs ridge, peak MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MT_059 | 239 kW | 70 kW / 140 kWh | 12.0% | 72% | 6% | 10% | $5,909 | $835 | 19 vs 23 kW |
| MT_061 | 159 kW | 55 kW / 110 kWh | 11.2% | 90% | 23% | 32% | $4,494 | $989 | 9 vs 9 kW |
| MT_128 | 772 kW | 235 kW / 470 kWh | 16.8% | 47% | 19% | 39% | $10,802 | $8,383 | 70 vs 80 kW |
| MT_151 | 289 kW | 85 kW / 170 kWh | 6.9% | 48% | 33% | 40% | $2,201 | $717 | 16 vs 15 kW |
| MT_158 | 215 kW | 65 kW / 130 kWh | 21.0% | 93% | 26% | 29% | $5,996 | $1,460 | 16 vs 15 kW |
| MT_164 | 262 kW | 80 kW / 160 kWh | 18.6% | 66% | 15% | 17% | $3,975 | $1,790 | 13 vs 12 kW |

<sub>Generated by `make report` from `out/results.json` (2025-01-01 to 2025-12-31, evaluation runtime 10.3s). Edit the code, not these numbers.</sub>
<!-- results:end -->

## What the numbers say

**The thesis holds in its general form and fails in its specific one.** In every case run here, annual savings spread further across dispatch policies than across forecasters; the robustness tables above give the margin, and show where it narrows (a large battery makes the two sensible policies close). But the policy the experiment started with, a threshold set as low as the forecast says the battery can hold, is the *worst* of the three, not the most graceful. It takes the forecast at its word and spends the whole battery on the predicted peak. When the real peak is higher, the battery is empty by the time it arrives. And because the demand charge bills the month's single worst interval, one bad day erases the month.

**Graceful depends on which error your policy is exposed to.** A meter-driven policy is immune to timing error but exposed to magnitude error (a target that is too low drains it early). A schedule is the reverse: bounded by its plan, so it degrades gently when the level is off, but blind to anything that arrives at a time it didn't plan for. On a building whose monthly peaks come from short equipment events, reading the meter and respecting the bill's structure (the ratchet) wins clearly. Take the spikes out and the schedule wins. A controller should be chosen for the errors its building actually produces.

**The real meters look like the spiky case, not the smooth one.** The worry with a synthetic study is that I built the data that produces the answer, so six real buildings were picked by a rule fixed in advance and run through the same pipeline. The ratchet came out on top on all of them, and none behaved like the smooth twin where the schedule wins. Real loads are also much harder to forecast than the synthetic one, which is the case where a policy that survives being wrong matters most. Six buildings in one country is a check, not a proof.

**The timing finding, plainly.** If the battery discharges on a schedule, a forecast that has the shape exactly right but is 15 minutes off in time loses most of the value on this building, because the peaks that set the bill are 15 to 45 minutes long. If the battery discharges when the meter crosses a level, timing error costs nothing; only the forecast's magnitude reaches it. With real forecasts, which are already blurred in time, the window policy's problem is the spikes it can't place rather than the clock.

**Average accuracy is the wrong yardstick.** The forecasters with the best business-day MAPE were not the ones with the best day-ahead peak error, and neither metric predicted the dollar ranking well. That held on real data too: TimesFM's lower MAPE did not reliably turn into better peaks or more money. What matters is the error on the few intervals that set each month's bill, and whether the policy survives being wrong about them.

**What would move the money next:** intraday re-forecasting (re-solving the target as the day unfolds), an energy reserve sized to each forecaster's recent peak error, and targets set from the upper quantiles of a probabilistic forecast. TimesFM already returns them; the 10th and 90th percentiles are saved in the cache.

## Limitations

- **Mostly synthetic data.** The main experiment is one generated building and one random seed, with a spike process I chose, which is why the spike-free twin is reported alongside. The six real meters are a check: anonymous clients, one country, one year, and an MA tariff applied to Portuguese buildings. MT_151 has a day of zero readings and MT_164 a few; they were left in, as real data. No confidence intervals: forecaster differences of a few percentage points are within noise.
- **Illustrative tariff.** No customer charges, demand ratchets, coincident-peak or capacity charges, and no real utility's filed rates.
- **No degradation model.** Cycling is free, so the policies that cycle more are not penalised for it.
- **No grid-services revenue.** No demand response, frequency regulation or wholesale arbitrage stacked on bill savings.
- **Single meter, single battery,** no portfolio effects.
- **Day-ahead only.** No intraday updates, which any production controller would add.
- **TimeGPT** is implemented and tested but only runs with a `NIXTLA_API_KEY`. Without one it drops out of the comparison rather than being estimated (see the results section for whether it ran).
- **TimesFM 3.0 weights** are licensed for non-commercial use only. TimesFM 2.5 weights are Apache-2.0, which is why both are here.

## Run it

```bash
make install     # core deps; no torch needed, foundation-model forecasts come from out/cache
make run         # evaluate everything -> out/results.json + tidy CSVs (~30 s on a laptop CPU)
make real        # the same on the six real meters -> out/real/ (~30 s, TimesFM from cache)
make test        # pytest
make report      # out/report.html, docs/index.html, and the README results above
make app         # Streamlit dashboard
```

On Windows without `make`, run the underlying commands: `pip install -r requirements.txt`, `python -m src.evaluate`, `python -m src.real_meters`, `python -m pytest -q`, `python -m src.report`, `streamlit run app.py`.

Outputs in `out/`: `results.json` (everything except per-day rows), plus tidy CSVs `summary`, `timing_sweep`, `daily`, `error_slopes`, `monthly`, `forecast_accuracy`, `battery_sensitivity`. Each has a `building` column. Nothing downstream (dashboard, report, README) types in a number; all of it is read from these files or recomputed by the same functions.

## Use your own meter data

Change one line in [`src/config.py`](src/config.py):

```python
DATA_SOURCE = "data/my_building_15min.csv"
```

`load_interval_csv()` reads a utility interval export or an ISO-NE load file into the same shape as the synthetic data: a 15-minute DatetimeIndex and one `kw` column. It auto-detects the timestamp and value columns, converts kWh or MW(h) to kW, maps any interval length onto 15 minutes, converts UTC offsets to local time (default `America/New_York`), and handles the DST hours. The evaluation uses the last 365 complete days that have 35 days of history before them. The no-spike counterfactual only exists for synthetic data.

## Foundation models

**TimesFM** (`TimesFMZeroShot` / `timesfm_zeroshot()` in `src/foundation.py`) runs Google's checkpoint locally on CPU through the current `timesfm` package (3.0.x): `TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cpu")`, then `predict_batch(contexts, horizon=96)` with a 28-day trailing context, batched 32 days at a time. Version 2.5 uses `TimesFM_2p5_200M_torch` with a compiled `ForecastConfig`. A year of day-ahead forecasts took about five minutes per model on an 8-core laptop CPU when the cache was built.

```bash
make install-fm   # pip install "timesfm[torch]" nixtla
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
- the forecast cache key, and TimeGPT skipping cleanly without a key
- `bill()`: the demand charge is the month's highest interval times the rate

## Layout

```
src/
  config.py      data source switch, paths
  data.py        synthetic building + load_interval_csv()
  forecast.py    forecasters (oracle, seasonal naive, profile, ridge) and the day-ahead runner
  foundation.py  TimesFM and TimeGPT, zero-shot, with the forecast cache
  tariff.py      Tariff dataclass and bill()
  dispatch.py    Battery, daily target bisection, simulate()
  evaluate.py    forecaster x policy x timing grid -> out/
  charts.py      plotly figures shared by the dashboard and report
  report.py      out/report.html, docs/index.html, README results
  uci.py         download + clean the UCI meters, pick buildings by a fixed rule
  real_meters.py the same experiment on the real meters -> out/real/
app.py           Streamlit dashboard
data/uci/        six cleaned real meters + how they were picked
tests/           pytest
out/             results, CSVs, report, forecast cache (committed)
```

MIT licence.
