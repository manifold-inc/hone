#!/bin/bash

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CURRENT_USER=$(whoami)

echo "Installing Hone validator auto-update service..."
echo "Repository path: $REPO_DIR"
echo "User: $CURRENT_USER"

chmod +x "${REPO_DIR}/validator/autoupdate/update.sh"
chmod +x "${REPO_DIR}/validator/autoupdate/service.sh"

mkdir -p "${REPO_DIR}/validator/logs/autoupdate"

SERVICE_FILE="/tmp/validator-updater.service"
cp "${REPO_DIR}/validator/autoupdate/validator-updater.service.template" "$SERVICE_FILE"

sed -i "s|%%USER%%|${CURRENT_USER}|g" "$SERVICE_FILE"
sed -i "s|%%REPO_DIR%%|${REPO_DIR}|g" "$SERVICE_FILE"

echo "Installing systemd service..."
sudo mv "$SERVICE_FILE" "/etc/systemd/system/hone-validator-updater.service"
sudo systemctl daemon-reload

echo "Enabling and starting service..."
sudo systemctl enable hone-validator-updater.service
sudo systemctl start hone-validator-updater.service

echo "Setting up log rotation..."
LOGROTATE_FILE="/etc/logrotate.d/hone-validator-updater"
sudo bash -c "cat > $LOGROTATE_FILE" << EOL
${REPO_DIR}/validator/logs/autoupdate/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 ${CURRENT_USER} ${CURRENT_USER}
}
EOL

echo "Installation complete!"
echo ""
echo "To check service status:"
echo "  sudo systemctl status hone-validator-updater.service"
echo ""
echo "To view update logs:"
echo "  tail -f ${REPO_DIR}/validator/logs/autoupdate/update.log"
echo ""
echo "To view service logs:"
echo "  tail -f ${REPO_DIR}/validator/logs/autoupdate/service.log"
echo "  sudo journalctl -u hone-validator-updater.service -f"