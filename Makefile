.PHONY: install test demo run

install:
	pip install -r requirements.txt

test:
	pytest -q

demo:
	streamlit run app.py

run:
	python scripts/run_experiment.py
