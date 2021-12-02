#!/bin/bash

exp_dir="./exp"
tag=""

n_sources=2
sr_k=8 # sr_k=8 means sampling rate is 8kHz. Choose from 8kHz or 16kHz.
sample_rate=${sr_k}000
duration=3.2 # 25600 samples
max_or_min='min'

test_wav_root="../../../dataset/wsj0-mix/${n_sources}speakers/wav${sr_k}k/${max_or_min}/tt"
test_list_path="../../../dataset/wsj0-mix/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_tt_mix"

window_fn='hann'
n_fft=256
hop_length=64
ideal_mask='ibm'
threshold=40
target_type='source'

# Embedding dimension
K=20

# Network configuration
H=300
B=4
dropout=0
causal=0
mask_nonlinear='sigmoid'
iter_clustering=-1
take_log=1
take_db=0

# Criterion
criterion='se' # or 'l2loss'

# Optimizer
optimizer='rmsprop'
lr=1e-4
weight_decay=0
max_norm=0 # 0 is handled as no clipping

batch_size=64
epochs_train=150
epochs_finetune=150

finetune=1 # If you don't want to use fintuned model, set `finetune=0`.
model_choice="last"

use_cuda=1
compute_attractor=1
estimate_all=1
overwrite=0
seed=111
gpu_id="0"

. ./path.sh
. parse_options.sh || exit 1

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/${n_sources}mix/sr${sr_k}k_${max_or_min}/${duration}sec/${criterion}/${target_type}"
    save_dir="${save_dir}/stft${n_fft}-${hop_length}_${window_fn}-window/${ideal_mask}_threshold${threshold}/K${K}_H${H}_B${B}_dropout${dropout}_causal${causal}_mask-${mask_nonlinear}"
    if [ ${take_log} -eq 1 ]; then
        save_dir="${save_dir}/take_log"
    elif [ ${take_db} -eq 1 ]; then
        save_dir="${save_dir}/take_db"
    else
        save_dir="${save_dir}/take_identity"
    fi
    if [ ${finetune} -eq 1 ]; then
        save_dir="${save_dir}/b${batch_size}_e${epochs_train}+${epochs_finetune}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}"
    else
        save_dir="${save_dir}/b${batch_size}_e${epochs_train}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}"
    fi
    save_dir="${save_dir}/seed${seed}"
else
    save_dir="${exp_dir}/${tag}"
fi

wrapper_save_dir="${save_dir}/fixed-attractor"
base_model_path="${save_dir}/model/${model_choice}.pth"
wrapper_model_dir="${wrapper_save_dir}/model"
log_dir="${wrapper_save_dir}/log"
out_dir="${wrapper_save_dir}/test"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="${gpu_id}"

test_fixed.py \
--test_wav_root "${test_wav_root}" \
--test_list_path "${test_list_path}" \
--sample_rate ${sample_rate} \
--window_fn ${window_fn} \
--ideal_mask ${ideal_mask} \
--threshold ${threshold} \
--target_type ${target_type} \
--n_fft ${n_fft} \
--hop_length ${hop_length} \
--iter_clustering ${iter_clustering} \
--n_sources ${n_sources} \
--criterion ${criterion} \
--out_dir "${out_dir}" \
--base_model_path "${base_model_path}" \
--wrapper_model_dir "${wrapper_model_dir}" \
--use_cuda ${use_cuda} \
--compute_attractor ${compute_attractor} \
--estimate_all ${estimate_all} \
--overwrite ${overwrite} \
--seed ${seed} | tee "${log_dir}/test_${time_stamp}.log"
