#!/bin/bash
# test_local.sh

echo "🚀 Starting Hone Subnet Local Test Environment"
echo "=============================================="

# Clean up any existing containers
echo "📦 Cleaning up existing containers..."
docker-compose -f validator/docker-compose.test.yml down -v

# Build images
echo "🔨 Building Docker images..."
docker-compose -f validator/docker-compose.test.yml build

# Start database first
echo "🗄️ Starting database..."
docker-compose -f validator/docker-compose.test.yml up -d db

# Wait for database
echo "⏳ Waiting for database to be ready..."
sleep 5

# Start miners
echo "⛏️ Starting mock miners..."
docker-compose -f validator/docker-compose.test.yml up -d miner1 miner2 miner3

# Wait for miners to be ready
echo "⏳ Waiting for miners to initialize..."
sleep 5

# Start validator (attached so we can see logs)
echo "✅ Starting validator..."
echo "=============================================="
echo "Press Ctrl+C to stop all services"
echo "=============================================="
docker-compose -f validator/docker-compose.test.yml up validator

# Cleanup on exit
echo "🧹 Cleaning up..."
docker-compose -f validator/docker-compose.test.yml down