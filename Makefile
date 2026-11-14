.PHONY: install data test reproduce app api docker compose report clean
PY ?= python

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/download_data.py

test:
	PYTHONPATH=src $(PY) -m pytest -q tests

reproduce:
	bash run_all.sh

app:
	PYTHONPATH=src streamlit run app/streamlit_app.py

api:
	PYTHONPATH=src uvicorn app.api:app --port 8000

docker:
	docker build -t rag-ecc .

compose:
	docker compose up --build

report:
	cd report && xelatex rapport.tex && xelatex rapport.tex
