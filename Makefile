test:
	python -m pytest -vv --cov-report term-missing --no-cov-on-fail --cov=tf-explain/ --timeout 10

lint:
	pylint tf-explain/
