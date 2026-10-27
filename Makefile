.PHONY: setup test demo real db-up db-down notebook clean

PY ?= python

setup:            ## install dependencies
	$(PY) -m pip install -r requirements.txt

test:             ## run the unit + integration tests (30 pass; the PostGIS test needs TEST_DATABASE_URL)
	$(PY) -m pytest -q

demo:             ## full pipeline on SYNTHETIC data (a software test, not a result)
	$(PY) -m highveld_co2.synthetic --days 730
	$(PY) -m highveld_co2.pipeline --raw data/raw/synthetic --out results_demo

real:             ## full pipeline on REAL OCO-2 files in data/raw/oco2 -> results/
	$(PY) -m highveld_co2.pipeline --raw data/raw/oco2 --out results

db-up:            ## start PostGIS in Docker
	docker compose up -d

db-down:
	docker compose down

notebook:         ## open the teaching notebook
	jupyter notebook notebooks/01_walkthrough.ipynb

clean:
	rm -rf results_demo data/raw/synthetic .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
