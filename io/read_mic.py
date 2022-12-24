
import pyaudio
import wave

CHUNK = 1024
WIDTH = 2
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 6
dev_index=1
WAVE_OUTPUT_FILENAME = 'test_audio_12.wav'
form_1 = pyaudio.paInt16
p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input_device_index=dev_index,
                input=True,
                output=False,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(form_1))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()