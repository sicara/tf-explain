test:
	python -m pytest -vv --cov-report term-missing --no-cov-on-fail --cov=mentat/ --timeout 10

lint:
	pylint mentat/
