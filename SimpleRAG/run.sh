#!/bin/bash
# Quick start script
echo "Starting SimpleRAG..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
