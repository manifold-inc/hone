source miner/.venv/bin/activate

pm2 start --name sn5-1 "python -m miner.main --port 5001 --wallet-name sn5 --wallet-hotkey h1"
pm2 start --name sn5-2 "python -m miner.main --port 5002 --wallet-name sn5 --wallet-hotkey h2"
pm2 start --name sn5-3 "python -m miner.main --port 5003 --wallet-name sn5 --wallet-hotkey h3"
pm2 start --name sn5-4 "python -m miner.main --port 5004 --wallet-name sn5 --wallet-hotkey h4"
pm2 start --name sn5-5 "python -m miner.main --port 5005 --wallet-name sn5 --wallet-hotkey h5"
