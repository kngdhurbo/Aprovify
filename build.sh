#!/usr/bin/env bash
# Render Build Script for Aprovify

set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Run database migrations
python manage.py migrate

# Train the ML model (generates model_robust.pkl)
python predictor_app/train_model.py

# Collect all static files into STATIC_ROOT for WhiteNoise to serve
python manage.py collectstatic --no-input
