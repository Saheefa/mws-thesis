#!/usr/bin/env bash
# Download the Chakraborty et al. (2016) Clickbait dataset into data/raw/.
# Usage (from the repo root):  bash scripts/get_clickbait_data.sh
set -euo pipefail

REPO=https://github.com/bhargaviparanjape/clickbait
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

git clone --depth 1 "$REPO" "$TMP/clickbait"
mkdir -p data/raw
gunzip -c "$TMP/clickbait/dataset/clickbait_data.gz"     > data/raw/clickbait_data
gunzip -c "$TMP/clickbait/dataset/non_clickbait_data.gz" > data/raw/non_clickbait_data
wc -l data/raw/clickbait_data data/raw/non_clickbait_data
