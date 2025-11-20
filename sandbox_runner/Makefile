install:
	pip3 install -r requirements.txt

setup:
	@echo "Running system setup..."
	sudo bash setup.sh

run:
	pm2 start main --name hone-runner --interpreter python3 -- --config config.local.yaml

dash:
	pm2 start dash_pannel.py --name dash-panel --interpreter python3

