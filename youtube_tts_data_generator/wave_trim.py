import json
import wave

# Specify the input WAV file path
input_wav_file = '/jREG_prep/downloaded/jREG1.wav'

# Specify the JSON file path containing the timestamps
timestamps_file = '/jREG_prep/downloaded/jREG1.en.json'

# Specify the output directory to save the trimmed clips
output_directory = 'jREG/'

def trim_wav(wav: wave.Wave_read, output_file, start_time, end_time):
    framerate = wav.getframerate()
    start_frame = int(start_time * framerate)
    end_frame = int(end_time * framerate)
    n_frames = end_frame - start_frame

    wav.setpos(start_frame)
    frames = wav.readframes(n_frames)

    with wave.open(output_file, 'wb') as trimmed_wav:
        trimmed_wav.setparams(wav.getparams())
        trimmed_wav.writeframes(frames)

def trim_wav_with_timestamps(input_wav, timestamps_file: str, output_directory: str, file_prefix: str = 'clip'):
    with open(timestamps_file, 'r') as file:
        timestamps = json.load(file)
    with wave.open(input_wav, 'rb') as wav:
        for i, timestamp in enumerate(timestamps, 1):
            start_time = timestamp['start']
            end_time = timestamp['end']
            output_file = f'{output_directory}/{file_prefix}-{i}.wav'

            trim_wav(wav, output_file, start_time, end_time)
            print(f'Trimmed clip {i} saved as {output_file}')

if __name__ == '__main__':
    trim_wav_with_timestamps(input_wav_file, timestamps_file, output_directory)