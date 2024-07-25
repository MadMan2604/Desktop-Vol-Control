import pyaudio
import numpy as np
import time

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
CLAP_THRESHOLD = 5000  # Adjust this threshold based on your environment
CLAP_TIME_LIMIT = 2  # Maximum time in seconds between claps for them to be considered consecutive
CLAP_DURATION = 0.1  # Maximum duration for a clap in seconds

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

clap_count = 0
last_clap_time = 0
clap_start_time = None

def detect_clap(data):
    """Detects if a clap sound is present based on the audio data."""
    audio_data = np.frombuffer(data, dtype=np.int16)
    peak = np.max(audio_data)
    if peak > CLAP_THRESHOLD:
        global clap_start_time
        if clap_start_time is None:
            clap_start_time = time.time()  # Start timing the clap duration
        elif time.time() - clap_start_time > CLAP_DURATION:
            return True
    else:
        clap_start_time = None
    return False

try:
    print("Listening for claps...")
    while True:
        # Read audio data
        data = stream.read(CHUNK)
        
        # Detect clap
        if detect_clap(data):
            current_time = time.time()
            if current_time - last_clap_time < CLAP_TIME_LIMIT:
                clap_count += 1
            else:
                clap_count = 1  # Reset if claps are too far apart
            last_clap_time = current_time
            
            # Check for double clap
            if clap_count == 2:
                print("Double clap detected!")
                clap_count = 0  # Reset count after detecting a double clap
        
except KeyboardInterrupt:
    print("Stopping clap detection.")
    
finally:
    # Stop and close audio stream
    stream.stop_stream()
    stream.close()
    p.terminate()
