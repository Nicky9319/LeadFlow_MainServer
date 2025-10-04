#!/bin/bash

set -e  # Exit on any error

echo "🚀 Deploying to production..."

# Define target directory
TARGET_DIR="/home/azureuser/backend"

# Clean old deployment
sudo -S rm -rf $TARGET_DIR  # Use -rf to avoid prompt and handle directories properly
sudo -S mkdir -p $TARGET_DIR

# Copy new files
sudo -S cp -r ./* $TARGET_DIR/

# Change ownership (optional, but safer if other services access this folder)
sudo -S chown -R paarth:paarth $TARGET_DIR

# Move into the target directory
cd $TARGET_DIR

# Installing the latest version of Python 3.12 venv
sudo -S apt install python3.12-venv -y
sudo -S apt install -y python3.12-dev build-essential

# Define environmental variables
python3.12 -m venv .venv

# Activate the virtual environment and install dependencies
source .venv/bin/activate

# Define environmental variables
pip install -r requirements.txt

# Ensure .env file exists and add a value
touch .env
echo -e "$ENVIRONMENTAL_VARIABLES" > .env
sed -i 's/\r//g' .env

# Making the logs Directory
rm -rf logs
mkdir logs

pm2 stop all || echo "⚠️ PM2 stop failed, continuing..."
pm2 flush
pm2 start process.json

sudo -S docker-compose down --remove-orphans || echo "⚠️ docker-compose down failed, continuing..."
sudo -S docker-compose up -d 

echo "✅ Deployment complete."



