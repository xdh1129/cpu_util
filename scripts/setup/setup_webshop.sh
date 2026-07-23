#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
revision=64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd

command -v java >/dev/null || {
  echo "Install OpenJDK 17 first: sudo apt-get install -y openjdk-17-jdk" >&2
  exit 1
}
python3 -m venv .venv-webshop
.venv-webshop/bin/python -m pip install --upgrade pip setuptools wheel
.venv-webshop/bin/python -m pip install -r requirements-webshop.txt
.venv-webshop/bin/python -m pip install --no-deps pyserini==0.17.0
.venv-webshop/bin/python -m pip install \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

if [[ ! -d .cache/webshop/.git ]]; then
  git clone https://github.com/princeton-nlp/WebShop.git .cache/webshop
fi
git -C .cache/webshop fetch origin "$revision"
git -C .cache/webshop checkout --detach "$revision"
HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/data/prepare_webshop.py
