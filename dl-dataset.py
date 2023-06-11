import os

import scrapetube
from youtube_tts_data_generator import YTSpeechDataGenerator

# Find youtube channel ID from youtube channel URL
# Example: https://www.youtube.com/channel/@jREG
# Channel ID: UCGGvjs7NQEkWqZYEoIJEG5g
youtuber_name = 'jREG'

channel_id = scrapetube.get_channel_id(youtuber_name)

MAX_VIDEOS = 30

links = scrapetube.get_video_links(os.environ["YOUTUBE_API_KEY"], channel_id)[:MAX_VIDEOS]

open(f'{youtuber_name}-links.txt', 'w').write('\n'.join(links))
# First create a YTSpeechDataGenerator instance:

generator = YTSpeechDataGenerator(dataset_name=youtuber_name, keep_audio_extension=True)

# Now create a '.txt' file that contains a list of YouTube videos that contains speeches.
# NOTE - Make sure you choose videos with subtitles.

generator.prepare_dataset(f'{youtuber_name}-links.txt', sr=44100)
# The above will take care about creating your dataset, creating a metadata file and trimming silence from the audios.

# Zip the dataset folder and upload it to your Google Drive.
# Now you can use the dataset in your Google Colab notebook.
generator.zip_dataset(f'{youtuber_name}-dataset')

# use S3 to upload the zip file to your S3 bucket
s3_path = f's3://eva-tts-datasets/'

import boto3

s3 = boto3.client('s3')

# Upload the zip file to S3


s3.upload_file(s3_path, 'eva-tts-datasets', f'{youtuber_name}-dataset.zip')

# # generator.finalize_dataset(min_audio_length=3, max_audio_length=30)
