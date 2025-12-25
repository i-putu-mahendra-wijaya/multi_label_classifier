requirements:
	# backup existing requirements.txt
	cp requirements.txt requirements.txt.swp;

	# create new requirements.txt
	python3 -m pip list --format=freeze > requirements.txt;

config_yaml:
	# run config_creator to automatically create a new config.yaml
	python3 config_creator.py