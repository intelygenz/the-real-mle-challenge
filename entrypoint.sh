#!/bin/bash

set -e

if [ "$1" = 'api' ]; then
    exec uvicorn api:app --host 0.0.0.0 --port 8000
elif [ "$1" = 'train' ]; then
    exec python training/main.py
elif [ "$1" = 'test' ]; then
    exec python -m unittest discover -s tests
else
    exec "$@"
fi