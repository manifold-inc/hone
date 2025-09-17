setup:
	@cp -n validator/.env.example validator/.env 2>/dev/null || echo "validator/.env exists"
	@cp -n miner/.env.example miner/.env 2>/dev/null || echo "miner/.env exists"

val:
	cd validator && docker compose down -v && docker compose up --build

down:
	cd validator && docker-compose down -v