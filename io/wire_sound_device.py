#!/usr/bin/env python3
"""Pass input directly to output.

https://github.com/PortAudio/portaudio/blob/master/test/patest_wire.c

python wire_sound_device.py -i 0 -o 2 --samplerate 44100 --channels 1 --dtype int16 (or float32) works
"""
import os
import sys
import argparse
import torch
import torchaudio
import sounddevice as sd
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)

sys.path.append("DNNSS/src")
#os.chdir("DNNSS/src")
print(os.getcwd())

from DNNSS.src.models.conv_tasnet import ConvTasNet
SAMPLE_RATE_WSJ0 = 8000
n_sources=2
model = ConvTasNet.build_from_pretrained(task="wsj0-mix", sample_rate=SAMPLE_RATE_WSJ0, n_sources=n_sources)
waveform_aew, sample_rate = torchaudio.load("C:/users/raghu/test/test_audio_11.wav")
resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE_WSJ0)
waveform_aew = resampler(waveform_aew)
waveform_aew = torch.reshape(waveform_aew, (waveform_aew.numel(),1))


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    '-i', '--input-device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-o', '--output-device', type=int_or_str,
    help='output device (numeric ID or substring)')
parser.add_argument(
    '-c', '--channels', type=int, default=2,
    help='number of channels')
parser.add_argument('--dtype', help='audio data type')
parser.add_argument('--samplerate', type=float, help='sampling rate')
parser.add_argument('--blocksize', type=int, help='block size')
parser.add_argument('--latency', type=float, help='latency in seconds')
args = parser.parse_args(remaining)

position = 0
print(position)
def process_data(indata):
    global position
    	
    x = torch.from_numpy(indata)
    l = x.numel()
    if position + l > waveform_aew.numel():
    	position=0
    print(waveform_aew.size(), x.size())
    print(waveform_aew[11000:11000+l].size())
    x=x #+ 2.0*waveform_aew[position:position+l]
    position = position+l
    print(x.size())
    x=torch.reshape(x, (1,1,x.numel()))
    print(x.size())
    with torch.no_grad():
        output = model(x)

    output = torch.reshape(output,(2,output.size()[2]))
    return output[0]

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    
    print(indata.shape, indata.dtype)
    x = process_data(indata)
    print(x.size())
    x=torch.reshape(x,(x.numel(),1))
    outdata[:] = x.numpy()


try:
    with sd.Stream(device=(args.input_device, args.output_device),
                   samplerate=args.samplerate, blocksize=args.blocksize,
                   dtype=args.dtype, latency=args.latency,
                   channels=args.channels, callback=callback):
        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))