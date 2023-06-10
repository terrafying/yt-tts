import json
import os

import requests

API_KEY = os.environ["YOUTUBE_API_KEY"]
def get_video_links(api_key, channel_id):
    base_url = "https://www.googleapis.com/youtube/v3/"
    playlist_id = get_playlist_id(api_key, channel_id)
    video_links = []

    next_page_token = None
    while True:
        url = base_url + "playlistItems"
        params = {
            "part": "snippet",
            "maxResults": 50,
            "playlistId": playlist_id,
            "key": api_key,
            "pageToken": next_page_token
        }

        response = requests.get(url, params=params)
        data = json.loads(response.text)

        for item in data["items"]:
            video_id = item["snippet"]["resourceId"]["videoId"]
            video_url = "https://www.youtube.com/watch?v=" + video_id
            video_links.append(video_url)

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return video_links

def get_playlist_id(api_key, channel_id):
    base_url = "https://www.googleapis.com/youtube/v3/"
    url = base_url + "channels"
    params = {
        "part": "contentDetails",
        "id": channel_id,
        "key": api_key
    }

    response = requests.get(url, params=params)
    data = json.loads(response.text)
    playlist_id = data["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    return playlist_id


def get_channel_id(channel_name):
    url = f'https://www.googleapis.com/youtube/v3/search?key={API_KEY}&part=id&type=channel&q={channel_name}'

    response = requests.get(url)
    data = response.json()

    # Extract the channel ID from the response
    channel_id = data['items'][0]['id']['channelId']
    return channel_id



if __name__ == "__main__":
    CHANNEL_ID = "UCGGvjs7NQEkWqZYEoIJEG5g"

    video_links = get_video_links(API_KEY, CHANNEL_ID)

    # Print all video links
    for link in video_links:
        print(link)

