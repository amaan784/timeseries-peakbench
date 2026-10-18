# On Windows without make, run the commands after each target directly.
PY ?= python

.PHONY: install test install-fm run forecasts report

install:            ## core dependencies (no torch: foundation-model forecasts come from out/cache)
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

install-fm:         ## TimesFM (CPU torch) + Nixtla client, only needed to regenerate the cache
	$(PY) -m pip install -r requirements-fm.txt

run:                ## evaluate every forecaster x policy; writes out/results.json + CSVs (< 60 s on CPU)
	$(PY) -m src.evaluate

forecasts:          ## rerun the foundation models for any (model, date) missing from out/cache
	$(PY) -m src.evaluate

report:             ## out/report.html + docs/index.html, and refresh the README results section
	$(PY) -m src.report
