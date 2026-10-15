# On Windows without make, run the commands after each target directly.
PY ?= python

.PHONY: install test

install:            ## core dependencies (no torch: foundation-model forecasts come from out/cache)
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q
