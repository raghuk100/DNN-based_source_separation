#!/bin/bash

wham_noise_root="../../../dataset/wham_noise"

. ./parse_options.sh || exit 1

file="wham_noise.zip"

if [ -e "${wham_noise_root}/wham_noise/tr/40na010x_1.9857_01xo031a_-1.9857.wav" ] ; then
    echo "Already downloaded dataset ${wham_noise_root}"
else
    wget "https://storage.googleapis.com/whisper-public/${file}" -P "/tmp"
    unzip "/tmp/${file}" -d "/tmp/"
    mv "/tmp/wham_noise/" "${wham_noise_root}"
    rm "/tmp/${file}"
fi
