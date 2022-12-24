import os
import sys
import argparse
sys.path.append('.\\common')
sys.path.append('.\\local')
sys.path.insert(-1, '..\\..\\..\\src')
sys.path.insert(-1, '..\\common\\src')

import train

exp_dir="exp"
continue_from=""
tag=""

n_sources=2

wav_root="../../../dataset/LibriSpeech"
train_json_path="../../../dataset/LibriSpeech/train-clean-100/train-100-{n_sources}mix.json".format(n_sources=n_sources)
valid_json_path="../../../dataset/LibriSpeech/dev-clean/valid-{n_sources}mix.json".format(n_sources=n_sources)
print(train_json_path, valid_json_path)
sample_rate=16000

# Encoder & decoder
enc_basis='trainable' # choose from ['trainable','Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']
dec_basis='trainable' # choose from ['trainable','Fourier', 'trainableFourier', 'trainableFourierTrainablePhase', 'pinv']
enc_nonlinear='relu' # enc_nonlinear is activated if enc_basis='trainable' and dec_basis!='pinv'
window_fn='' # window_fn is activated if enc_basis or dec_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']
enc_onesided=0 # enc_onesided is activated if enc_basis or dec_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']
enc_return_complex=0 # enc_return_complex is activated if enc_basis or dec_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']

N=64
L=16

# Separator
H=256
B=128
Sc=128
P=3
# Defaults here
X=6
R=3
#X=4
#R=2
dilated=1
separable=1
causal=1
sep_nonlinear='prelu'
sep_norm=1
use_batch_norm=0
mask_nonlinear='sigmoid'

# Criterion
criterion='sisdr'

# Optimizer
optimizer='adam'
lr=1e-3
weight_decay=0
max_norm=5

batch_size=4
epochs=100

use_cuda=1
overwrite=0
seed=111



prefix=""



save_dir="{exp_dir}\\{n_sources}mix\\{enc_basis}-{dec_basis}\\{criterion}\\" \
          "N{N}_L{L}_B{B}_H{H}_Sc{Sc}_P{P}_X{X}_R{R}\\" \
          "{prefix}dilated{dilated}_separable{separable}_causal{causal}_{sep_nonlinear}_norm{sep_norm}_bn{use_batch_norm}_mask-{mask_nonlinear}\\" \
          "b{batch_size}_e{epochs}_{optimizer}-lr{lr}-decay{weight_decay}_clip{max_norm}\\seed{seed}".format(exp_dir=exp_dir, n_sources=n_sources, enc_basis=enc_basis, dec_basis=dec_basis,
           criterion=criterion, N=N, L=L, B=B, H=H, Sc=Sc, P=P, X=X, R=R,
           prefix=prefix, dilated=dilated, separable=separable, causal=causal, sep_nonlinear=sep_nonlinear,
           sep_norm=sep_norm, use_batch_norm=use_batch_norm, mask_nonlinear=mask_nonlinear,
           batch_size=batch_size, epochs=epochs, optimizer=optimizer, lr=lr, weight_decay=weight_decay, max_norm=max_norm, seed=seed)
 
save_dir1="{exp_dir}\\{n_sources}mix\\{enc_basis}-{dec_basis}\\{criterion}\\" \
          "N{N}_L{L}_B{B}_H{H}_Sc{Sc}_P{P}_X{X}_R{R}\\" \
          "{prefix}dilated{dilated}_separable{separable}_causal{causal}_{sep_nonlinear}_norm{sep_norm}_bn{use_batch_norm}_mask-{mask_nonlinear}\\".format(exp_dir=exp_dir, n_sources=n_sources, enc_basis=enc_basis, dec_basis=dec_basis,
           criterion=criterion, N=N, L=L, B=B, H=H, Sc=Sc, P=P, X=X, R=R,
           prefix=prefix, dilated=dilated, separable=separable, causal=causal, sep_nonlinear=sep_nonlinear,
           sep_norm=sep_norm, use_batch_norm=use_batch_norm, mask_nonlinear=mask_nonlinear)

save_dir2="b{batch_size}_e{epochs}_{optimizer}-lr{lr}-d{weight_decay}_cl{max_norm}\\s{seed}".format(
           batch_size=batch_size, epochs=epochs, optimizer=optimizer, lr=lr, weight_decay=weight_decay, max_norm=max_norm, seed=seed)
#save_dir=save_dir.replace('.', '_')
save_dir1=save_dir1.replace('.', '_')
save_dir2=save_dir2.replace('.', '_')
from pathlib import Path
#Path(exp_dir).mkdir(parents=True, exist_ok=True)            
#print(save_dir)
Path(save_dir1).mkdir(parents=True, exist_ok=True) 
# Hack to handle long path names
curr_dir= os.getcwd()
os.chdir(save_dir1)
Path(save_dir2).mkdir(parents=True, exist_ok=True)


model_dir=save_dir2+"/model"
loss_dir=save_dir2+"/loss"
sample_dir=save_dir2+"/sample"
log_dir=save_dir2+"/log"
if not os.path.isdir(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)   
    print('Creating', log_dir)

# Change back to dir
os.chdir(curr_dir)
from datetime import datetime

# get current date
time_stamp = str(datetime.now())


parser = argparse.ArgumentParser(description="Training of Conv-TasNet")

parser.add_argument('--wav_root', type=str, default=None, help='Path for dataset ROOT directory')
parser.add_argument('--train_json_path', type=str, default=None, help='Path for train.json')
parser.add_argument('--valid_json_path', type=str, default=None, help='Path for valid.json')
parser.add_argument('--sample_rate', '-sr', type=int, default=8000, help='Sampling rate')
parser.add_argument('--enc_basis', type=str, default='trainable', choices=['trainable','Fourier','trainableFourier','trainableFourierTrainablePhase'], help='Encoder type')
parser.add_argument('--dec_basis', type=str, default='trainable', choices=['trainable','Fourier','trainableFourier','trainableFourierTrainablePhase', 'pinv'], help='Decoder type')
parser.add_argument('--enc_nonlinear', type=str, default=None, help='Non-linear function of encoder')
parser.add_argument('--window_fn', type=str, default='hann', help='Window function')
parser.add_argument('--enc_onesided', type=int, default=None, choices=[0, 1, None], help='If true, encoder returns kernel_size // 2 + 1 bins.')
parser.add_argument('--enc_return_complex', type=int, default=None, choices=[0, 1, None], help='If true, encoder returns complex tensor, otherwise real tensor concatenated real and imaginary part in feature dimension.')
parser.add_argument('--n_basis', '-N', type=int, default=512, help='# basis')
parser.add_argument('--kernel_size', '-L', type=int, default=16, help='Kernel size')
parser.add_argument('--stride', type=int, default=None, help='Stride. If None, stride=kernel_size//2')
parser.add_argument('--sep_bottleneck_channels', '-B', type=int, default=128, help='Bottleneck channels of separator')
parser.add_argument('--sep_hidden_channels', '-H', type=int, default=128, help='Hidden channels of separator')
parser.add_argument('--sep_skip_channels', '-Sc', type=int, default=128, help='Skip connection channels of separator')
parser.add_argument('--sep_kernel_size', '-P', type=int, default=3, help='Skip connection channels of separator')
parser.add_argument('--sep_num_layers', '-X', type=int, default=8, help='# layers of separator')
parser.add_argument('--sep_num_blocks', '-R', type=int, default=3, help='# blocks of separator. Each block has R layers')
parser.add_argument('--dilated', type=int, default=1, help='Dilated convolution')
parser.add_argument('--separable', type=int, default=1, help='Depthwise-separable convolution')
parser.add_argument('--causal', type=int, default=0, help='Causality')
parser.add_argument('--sep_nonlinear', type=str, default=None, help='Non-linear function of separator')
parser.add_argument('--sep_norm', type=int, default=1, help='Normalization')
parser.add_argument('--use_batch_norm', type=int, default=1, help='BN in seperator')
parser.add_argument('--mask_nonlinear', type=str, default='sigmoid', help='Non-linear function of mask estiamtion')
parser.add_argument('--n_sources', type=int, default=None, help='# speakers')
parser.add_argument('--criterion', type=str, default='sisdr', choices=['sisdr'], help='Criterion')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'], help='Optimizer, [sgd, adam, rmsprop]')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate. Default: 0.001')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty). Default: 0')
parser.add_argument('--max_norm', type=float, default=None, help='Gradient clipping')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size. Default: 128')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--model_dir', type=str, default='./tmp/model', help='Model directory')
parser.add_argument('--loss_dir', type=str, default='./tmp/loss', help='Loss directory')
parser.add_argument('--sample_dir', type=str, default='./tmp/sample', help='Sample directory')
parser.add_argument('--continue_from', type=str, default=None, help='Resume training')
parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
parser.add_argument('--overwrite', type=int, default=0, help='0: NOT overwrite, 1: FORCE overwrite')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

#train.main( myargs)
arg_str2= [
'--wav_root', wav_root, 
'--train_json_path', train_json_path,
'--valid_json_path', valid_json_path,
'--sample_rate', str(sample_rate),
'--enc_basis', enc_basis,
'--dec_basis', dec_basis,
'--enc_nonlinear', enc_nonlinear,
'--window_fn', window_fn,
'--enc_onesided', str(enc_onesided),
'--enc_return_complex', str(enc_return_complex),
'-N',str(N),
'-L',str(L),
'-B' ,str(B),
'-H' ,str(H),
'-Sc' ,str(Sc),
'-P' ,str(P),
'-X' ,str(X),
'-R' ,str(R),
'--dilated' ,str(dilated),
'--separable' ,str(separable),
'--causal' ,str(causal),
'--sep_nonlinear',str(sep_nonlinear),
'--sep_norm',str(sep_norm),
'--use_batch_norm',str(use_batch_norm),
'--mask_nonlinear',mask_nonlinear,
'--n_sources' ,str(n_sources),
'--criterion' ,criterion,
'--optimizer',optimizer,
'--lr',str(lr),
'--weight_decay',str(weight_decay),
'--max_norm',str(max_norm),
'--batch_size', str(batch_size),
'--epochs',str(epochs),
'--model_dir',model_dir,
'--loss_dir',loss_dir,
'--sample_dir', sample_dir,
'--continue_from', continue_from,
'--use_cuda', str(use_cuda),
'--overwrite', str(overwrite),
'--seed', str(seed)]

        
args = parser.parse_args(arg_str2)
print(args)
train.main( args)
#| tee "${log_dir}/train_${time_stamp}.log"
