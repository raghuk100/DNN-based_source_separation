#!/bin/bash

model_id="1GI1TEWsoMxlvVdKuLvP6aeau26fNy3Qb"

. ./path.sh
. parse_options.sh || exit 1

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${model_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${model_id}" -o "model.zip"

unzip "model.zip"