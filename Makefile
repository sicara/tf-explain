test:
	python -m pytest -vv --cov-report term-missing --no-cov-on-fail --cov=mentat/

lint:
	pylint mentat/
