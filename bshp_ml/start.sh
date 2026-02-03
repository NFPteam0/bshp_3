#!/bin/bash
set -e

if [ -z "$(ls -A /models)" ]; then
echo "No models found. Loading basic text model..."
mv /_models/* /models
fi

exec "$@"