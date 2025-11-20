install:
	pip3 install -r requirements.txt

setup:
	@echo "Running system setup..."
	sudo bash setup.sh

run:
	pm2 start "python3 -m main --config config.local.yaml" --name hone-runner

dash:
	pm2 start dash_pannel.py --name dash-panel --interpreter python3 --name dash
