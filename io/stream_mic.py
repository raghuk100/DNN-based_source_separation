
import pyaudio
import wave
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import time

def process_frame(signal_wave, b, a):
         
	# if one channel use int16, if 2 use int32
	audio_array = np.frombuffer(signal_wave, dtype=np.int16)
	out = np.sqrt(audio_array) #signal.lfilter(b,a,audio_array)
	#print(out)
	#print(out.type)
	z = out.astype(np.int16)
	print(z)
	return z.tobytes()
	times = np.linspace(0, 1, num=4096)

	plt.figure(figsize=(15, 5))
	plt.plot(audio_array)
	plt.ylabel('Signal Wave')
	plt.xlabel('Time (s)')
	#plt.xlim(0, time)
	plt.title('The Thing I Just Recorded!!')
	plt.show()
	
CHUNK = 4096
WIDTH = 2
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 30
input_dev_index=1
output_dev_index=2
#WAVE_OUTPUT_FILENAME = 'test_audio5.wav'
form_1 = pyaudio.paInt16
p = pyaudio.PyAudio()
for ii in range(p.get_device_count()):
	print(ii, p.get_device_info_by_index(ii).get('name'))

def callback(in_data, frame_count, time_info, flag):
 #   out = process_frame(in_data)
    print(frame_count)
  
    return (in_data, pyaudio.paContinue)
    
    
stream_mic = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input_device_index=input_dev_index,
                input=True,
                output=False,
                frames_per_buffer=CHUNK)
                
# open stream based on the wave object which has been input.
stream_out = p.open(format =
                p.get_format_from_width(WIDTH),
                channels = CHANNELS,
                rate = RATE,
                output_device_index=output_dev_index,
                output = True,
                input=False,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

print("* starting")

data=stream_mic.read(CHUNK)
out=data
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    print(i)
    data = stream_mic.read(CHUNK)
    time.sleep(1)
    #b = np.array([1.5])
    #a = np.array([1, 0.999])
  #  b, a = signal.butter(3, 0.01)
  #  b = 2.0*b
 
 

stream_mic.stop_stream()
stream_mic.close()
stream_out.stop_stream()
stream_out.close()
p.terminate()

