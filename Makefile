# On Windows without make, run the commands after each target directly.
PY ?= python

.PHONY: install install-fm run real uci test report app forecasts all

install:            ## core dependencies (no torch: foundation-model forecasts come from out/cache)
	$(PY) -m pip install -r requirements.txt

install-fm:         ## TimesFM (CPU torch) + Nixtla client, only needed to regenerate the cache
	$(PY) -m pip install -r requirements-fm.txt

run:                ## evaluate every forecaster x policy; writes out/results.json + CSVs (< 60 s on CPU)
	$(PY) -m src.evaluate

real:               ## same experiment on the six real UCI meters in data/uci/ -> out/real/
	$(PY) -m src.real_meters

uci:                ## re-download the UCI file and rebuild data/uci/ (250 MB download)
	$(PY) -m src.uci

test:
	$(PY) -m pytest -q

report:             ## out/report.html + docs/index.html, and refresh the README results section
	$(PY) -m src.report

app:                ## the Streamlit dashboard
	$(PY) -m streamlit run app.py

forecasts:          ## rerun the foundation models for any (model, date) missing from out/cache
	$(PY) -m src.evaluate

all: run real test report
