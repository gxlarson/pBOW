clean:
	find . -name "*.pyc" -exec rm -f {} \;

test:
	python -m unittest discover
