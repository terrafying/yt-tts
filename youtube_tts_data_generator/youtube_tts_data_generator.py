import errno
import json
import os
import re
import shutil
import warnings
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
import yt_dlp
from pydub import AudioSegment
from tqdm import tqdm
from vtt_to_srt.vtt_to_srt import convert_content
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
)
from youtube_transcript_api.formatters import JSONFormatter

from .audio import preprocess_wav
from .text_cleaner import Cleaner
from .wave_trim import trim_wav_with_timestamps

WAV_SUFFIX = ".wav"


class NoSubtitleWarning(UserWarning):
    pass


class YTLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)


def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)


def parse_time(time_string):
    hours = int(re.findall(r"(\d+):\d+:\d+,\d+", time_string)[0])
    minutes = int(re.findall(r"\d+:(\d+):\d+,\d+", time_string)[0])
    seconds = int(re.findall(r"\d+:\d+:(\d+),\d+", time_string)[0])
    milliseconds = int(re.findall(r"\d+:\d+:\d+,(\d+)", time_string)[0])

    return (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds


class YTSpeechDataGenerator(object):
    """
    YTSpeechDataGenerator makes it easier to
    generate data for Text to Speech/Speech to Text.  .

    Parameters:

    dataset_name:           Name of the dataset

    output_type:            Format of the metadata file.
                            Supported formats:(csv/json)

    keep_audio_extension:   Whether to keep the audio file name
                            extensions in the metadata file.

    lang:                   The target language of subtitles.

    Available methods:

    download:               Download wavs from YouTube from a .txt file.

    split_audios:           Split the downloaded single wav files into
                            multiple.

    concat_audios:          Merge multiple smaller audios into a bit
                            longer ones.

    finalize_dataset:       Generate final dataset from the processes
                            audios.

    get_available_langs:    Get list of available languages in which the
                            the subtitles can be downloaded.

    get_total_audio_length: Get the total length of audios.
    """

    def __init__(
            self,
            dataset_name,
            output_type="csv",
            keep_audio_extension=True,
            lang="en",
            sr=22050,
    ):
        self.wav_counter = 0
        self.wav_filenames = []
        self.name = dataset_name
        self.root = os.getcwd()
        self.prep_dir = os.path.join(self.root, self.name + "_prep")
        self.dest_dir = os.path.join(self.root, self.name)
        self.download_dir = os.path.join(self.prep_dir, "downloaded")
        self.split_dir = os.path.join(self.prep_dir, "split")
        self.concat_dir = os.path.join(self.prep_dir, "concatenated")
        self.filenames_txt = os.path.join(self.download_dir, "files.txt")
        self.split_audios_csv = os.path.join(self.split_dir, "split.csv")
        self.dataset_zip = os.path.join(self.root, self.name + ".zip")
        self.len_dataset = 0
        self.len_shortest_audio = 0
        self.len_longest_audio = 0
        self.keep_audio_extension = keep_audio_extension
        self.sr = sr
        if output_type not in ["csv", "json"]:
            raise Exception(
                "Invalid output type. Supported output files are 'csv'/'json'"
            )
        else:
            self.output_type = output_type
        self.cleaner = Cleaner()
        self.transcript_formatter = JSONFormatter()

        if not os.path.exists(self.prep_dir):
            print(f"Creating directory '{self.name}_prep'..")
            print(f"Creating directory '{self.name}_prep/downloaded'")
            print(f"Creating directory '{self.name}_prep/split'")
            print(f"Creating directory '{self.name}_prep/concatenated'")
            os.mkdir(self.prep_dir)
            os.mkdir(self.download_dir)
            os.mkdir(self.split_dir)
            os.mkdir(self.concat_dir)

        if not os.path.exists(self.dest_dir):
            print(f"Creating directory '{self.name}'..")
            print(f"Creating directory '{self.name}/wavs'")
            print(f"Creating directory '{self.name}/txts'")
            os.mkdir(self.dest_dir)
            os.mkdir(os.path.join(self.dest_dir, "wavs"))
            os.mkdir(os.path.join(self.dest_dir, "txts"))

        self.dataset_lang = lang

        self.ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "logger": YTLogger(),
        }

    def get_available_langs(self):
        print("List of supported languages:\n")
        for key, lang in self.lang_map.items():
            print(key, ":", lang)

    def get_video_id(self, url, pattern="(\/|%3D|v=)([0-9A-z-_]{11})([%#?&]|$)"):
        matches = re.findall(pattern, url)
        if matches != []:
            try:
                return matches[0][1]
            except:
                return []
        else:
            return []

    def fix_json_trans(self, trans):
        return [
            {
                "start": trans[ix]["start"],
                "end": trans[ix + 1]["start"],
                "text": trans[ix]["text"],
            }
            if ix != len(trans) - 1
            else {
                "start": trans[ix]["start"],
                "end": trans[ix]["start"] + trans[ix]["duration"],
                "text": trans[ix]["text"],
            }
            for ix in range(len(trans))
            if trans[ix]["text"] != "[Music]"
        ]

    def download(self, links_txt):
        """
        Downloads YouTube Videos as wav files.

        Parameters:
              links_txt: A .txt file that contains list of
                         youtube video urls separated by new line.
        """
        self.text_path = os.path.join(self.root, links_txt)
        if not os.path.isfile(self.text_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), links_txt)

        links = open(self.text_path).read().strip().split("\n")
        if not os.path.getsize(self.text_path) > 0:
            raise Exception(f"ERROR - File '{links_txt}' is empty")
        for i, link in enumerate(links):
            video_id = self.get_video_id(link)
            if not video_id:
                warnings.warn(
                    f"WARNING - video {link} does not seem to be a valid YouTube url. Skipping..",
                )
                continue

            filename = f"{self.name}{i + 1}.%(ext)s"

            wav_file = filename.rsplit('.', maxsplit=1)[0] + WAV_SUFFIX

            if os.path.isfile(os.path.join(self.download_dir, wav_file)):
                tqdm.write(f"{wav_file} exists. Skipping...")
                self.wav_filenames.append(wav_file)
                continue

            self.ydl_opts["outtmpl"] = os.path.join(
                self.download_dir, filename
            )

            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                try:
                    trans = (
                        YouTubeTranscriptApi.list_transcripts(video_id)
                        .find_transcript([self.dataset_lang])
                        .fetch()
                    )
                    trans = self.fix_json_trans(trans)
                    json_formatted = (
                        self.transcript_formatter.format_transcript(
                            trans, ensure_ascii=False, indent=2
                        )
                    )
                    open(
                        os.path.join(
                            self.download_dir,
                            wav_file.replace(
                                WAV_SUFFIX, f".{self.dataset_lang}.json"
                            ),
                        ),
                        "w",
                        encoding="utf-8",
                    ).write(json_formatted)
                    ydl.download([link])
                    print(
                        "Completed downloading "
                        + wav_file
                        + " from "
                        + link
                    )
                    self.wav_counter += 1
                    self.wav_filenames.append(wav_file)
                except (TranscriptsDisabled, NoTranscriptFound):
                    warnings.warn(
                        f"WARNING - video {link} does not have subtitles. Skipping..",
                        NoSubtitleWarning,
                    )
            del self.ydl_opts["outtmpl"]
        if self.wav_filenames:
            with open(self.filenames_txt, "w", encoding="utf-8") as f:
                lines = "filename,subtitle,trim_min_begin,trim_min_end\n"
                for wav in self.wav_filenames:
                    lines += f"{wav},{wav.rsplit('.', maxsplit=1)[0]}.{self.dataset_lang}.json,0,0\n"
                f.write(lines)
            print(f"Completed downloading audios to '{self.download_dir}'")
            print(f"You can find files data in '{self.filenames_txt}'")
        else:
            warnings.warn(
                f"WARNING - No video with subtitles found to create dataset.",
                NoSubtitleWarning,
            )

    def parse_srt(self, srt_string):
        # Original : https://github.com/pgrabovets/srt-to-json
        srt_list = []

        for line in srt_string.split("\n\n"):
            if line != "":
                index = int(re.match(r"\d+", line).group())

                pos = re.search(r"\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+", line).end() + 1
                content = line[pos:]
                start_time_string = re.findall(
                    r"(\d+:\d+:\d+,\d+) --> \d+:\d+:\d+,\d+", line
                )[0]
                end_time_string = re.findall(
                    r"\d+:\d+:\d+,\d+ --> (\d+:\d+:\d+,\d+)", line
                )[0]
                start_time = parse_time(start_time_string)
                end_time = parse_time(end_time_string)

                srt_list.append(
                    {
                        "text": content.replace("\n", "").strip(),
                        "start": start_time / 1000,
                        "duration": (end_time - start_time) / 1000,
                    }
                )

        return srt_list

    def split_audios(self):
        """
        Split the downloaded videos into smaller chunks.
        """
        if not os.path.isfile(self.filenames_txt):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), self.filenames_txt
            )

        files_list = open(self.filenames_txt).read().strip().split("\n")
        files_list = files_list[1:]
        tqdm.write(f"Found {len(files_list)} files to process")
        try:
            files_pbar = tqdm(files_list)
            for line in files_pbar:
                filename, subtitle, trim_min_begin, trim_min_end = line.split(",")
                caption_json = None
                json_caption_out_filename = filename.rsplit('.')[0] + f".en.json"
                files_pbar.set_description("Processing %s" % filename)
                if subtitle.lower().endswith(".vtt"):
                    tqdm.write(f"Detected VTT captions. Converting to json..")
                    file_contents = open(
                        os.path.join(self.download_dir, subtitle),
                        mode="r",
                        encoding="utf-8",
                    ).read()
                    srt = convert_content(file_contents)
                    caption_json = self.parse_srt(srt.strip())
                elif subtitle.lower().endswith(".srt"):
                    tqdm.write(f"Detected SRT captions. Converting to json..")
                    file_contents = open(
                        os.path.join(self.download_dir, subtitle),
                        mode="r",
                        encoding="utf-8",
                    ).read()
                    caption_json = self.parse_srt(file_contents.strip())
                elif subtitle.lower().endswith(".json"):
                    pass
                else:
                    raise Exception(
                        "Invalid subtitle type. Supported subtitle types are 'vtt'/'srt'"
                    )
                if caption_json:
                    caption_json = self.fix_json_trans(caption_json)
                    open(
                        os.path.join(self.download_dir, json_caption_out_filename),
                        "w",
                        encoding="utf-8",
                    ).write(json.dumps(caption_json, indent=2, sort_keys=True))
                    tqdm.write(
                        f"Writing json captions for {filename} to '{json_caption_out_filename}'."
                    )
                trim_min_end = int(trim_min_end)
                trim_min_begin = int(trim_min_begin)

                filename_stub = filename.rsplit(".", maxsplit=1)[0]

                if not caption_json:
                    with open(
                            os.path.join(self.download_dir, subtitle)
                    ) as json_cap:
                        captions = json.loads(json_cap.read())
                else:
                    captions = caption_json

                trim_min_end = captions[-1]["end"]
                # tqdm.write(f"Trim: {trim_min_end} {trim_min_begin}")
                # tqdm.write(str(captions))

                trim_wav_with_timestamps(
                    os.path.join(self.download_dir, filename),
                    os.path.join(self.download_dir, json_caption_out_filename),
                    self.split_dir,
                    filename_stub)

                for ix, caption in enumerate(captions):
                    with open(
                            os.path.join(self.split_dir, filename_stub + f"-{ix+1}.txt"), "w"
                    ) as f:
                        f.write(caption['text'])

            tqdm.write(
                f"Completed splitting audios and texts to '{self.split_dir}'"
            )

            files_pbar = tqdm(files_list)

            tqdm.write(f"Verifying split audios and their transcriptions.")

            df = []
            for line in files_pbar:
                filename, subtitle, trim_min_begin, trim_min_end = line.split(",")
                files_pbar.set_description("Processing %s" % filename)
                fname = filename[:-4]
                files = os.listdir(self.split_dir)
                wav_files = [f for f in files if f.endswith(WAV_SUFFIX)]
                for ix in range(len(wav_files)):
                    current_file = fname + '-' + str(ix) + ".txt"
                    current_wav = fname + '-' + str(ix) + WAV_SUFFIX
                    try:
                        current_text = (
                            open(os.path.join(self.split_dir, current_file))
                            .read()
                            .strip()
                        )

                        tqdm.write(f"Processing {current_file} - {current_text}")

                        wav, sr = librosa.load(
                            os.path.join(self.split_dir, current_wav)
                        )
                        length = wav.shape[0] / sr

                        if current_text != "" and length > 0.0:
                            df.append([current_wav, current_text, round(length, 2)])
                        else:
                            tqdm.write(f"Empty text or audio file. Skipping. {current_wav}")
                    except Exception as e:
                        print(e)

            df = pd.DataFrame(
                df, columns=["wav_file_name", "transcription", "length"]
            )

            df.to_csv(path_or_buf=Path(self.split_audios_csv), sep="|", index=False)

            tqdm.write(
                f"Completed verifying audios and their transcriptions in '{self.split_dir}'."
            )
            tqdm.write(f"You can find files data in '{self.split_audios_csv}'")

        except FileNotFoundError as e:
            tqdm.write(f"ERROR - {e}")

    def concat_audios(self, max_limit=30, concat_count=2):
        """
        Joins the chunk of audio files into
        audios of recognizable length.
        """
        if not os.path.isfile(self.split_audios_csv):
            tqdm.write(
                f"ERROR - Couldn't find file 'split.csv'. Make sure it is placed in {self.split_dir}"
            )
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), "split.csv"
            )
        tqdm.write(f"Reading audio data from '{self.split_audios_csv}'.")
        df = pd.read_csv(self.split_audios_csv, sep="|")
        filtered_df = df[df["length"] <= max_limit]
        long_audios = df[df["length"] > max_limit]

        name_ix = 0
        tqdm.write(f"Processing audios shorter than {max_limit} seconds..")
        for ix in tqdm(range(0, filtered_df.shape[0], 2)):
            current_audio = filtered_df.iloc[ix][0]
            text = ""
            combined_sounds = AudioSegment.from_wav(
                os.path.join(self.split_dir, current_audio)
            )
            text += " " + filtered_df.iloc[ix][1]
            try:
                for count_ix in range(ix + 1, ix + concat_count):
                    next_audio = filtered_df.iloc[count_ix][0]
                    sound2 = AudioSegment.from_wav(
                        os.path.join(self.split_dir, next_audio)
                    )
                    text += " " + filtered_df.iloc[count_ix][1]
                    combined_sounds += sound2

                text = text.strip()
                new_name = f"{self.name}-{name_ix}"
                combined_sounds.set_frame_rate(self.sr)
                combined_sounds.export(
                    os.path.join(self.concat_dir, new_name + WAV_SUFFIX), format="wav"
                )

                with open(
                        os.path.join(self.concat_dir, new_name + ".txt"), "w"
                ) as f:
                    f.write(text)
                name_ix += 1
            except IndexError:
                tqdm.write("Index error!")
                new_name = f"{self.name}-{name_ix}"
                combined_sounds = AudioSegment.from_wav(
                    os.path.join(self.split_dir, current_audio)
                )
                combined_sounds.set_frame_rate(self.sr)
                text = text.strip()
                combined_sounds.export(
                    os.path.join(self.concat_dir, new_name + WAV_SUFFIX), format="wav"
                )
                with open(
                        os.path.join(self.concat_dir, new_name + ".txt"), "w"
                ) as f:
                    f.write(text)
                name_ix += 1

        tqdm.write(f"Processing audios longer than {max_limit} seconds..")

        for ix in tqdm(range(0, long_audios.shape[0])):
            current_audio = filtered_df.iloc[ix][0]
            text = filtered_df.iloc[ix][1].strip()
            new_name = f"{self.name}-{name_ix}"
            combined_sounds = AudioSegment.from_wav(
                os.path.join(self.split_dir, current_audio)
            )
            combined_sounds.set_frame_rate(self.sr)
            combined_sounds.export(
                os.path.join(self.concat_dir, new_name + WAV_SUFFIX), format="wav"
            )
            with open(os.path.join(self.concat_dir, new_name + ".txt"), "w") as f:
                f.write(text)
            name_ix += 1

        tqdm.write(
            f"Completed concatenating audios and their transcriptions in '{self.concat_dir}'."
        )

    def get_total_audio_length(self):
        """
        Returns the total number of preprocessed audio
        in seconds.
        """

        tqdm.write(
            f"Collected {round(self.len_dataset / 3600, 2)}hours ({int(self.len_dataset)} seconds) of audio."
        )
        return int(self.len_dataset)

    def finalize_dataset(self, min_audio_length=3, max_audio_length=30):
        """
        Trims silence from audio files
        and creates a metadada file in csv/json format.

        Parameters:
            min_audio_length: The minimum length of audio files.

            max_audio_length: The maximum length of audio files.

            WAV_SUFFIX: .wav or .whatever
        """

        tqdm.write(f"Trimming silence from audios in '{self.concat_dir}'.")

        concat_audios = [
            wav for wav in os.listdir(self.concat_dir) if wav.endswith(WAV_SUFFIX)
        ]
        tqdm.write("concat_audios: " + str(concat_audios))

        # concat_txt = [wav.rsplit('.', maxsplit=1)[0] + ".txt" for wav in concat_audios]

        filtered_audios = []
        filtered_txts = []
        audio_lens = []

        for ix in tqdm(range(len(concat_audios))):
            audio = concat_audios[ix]
            wav, sr = librosa.load(os.path.join(self.concat_dir, audio))
            silence_removed = preprocess_wav(wav)
            trimmed_length = silence_removed.shape[0] / sr
            audio_lens.append(trimmed_length)

            if min_audio_length <= trimmed_length <= max_audio_length:
                self.len_dataset += trimmed_length
                """
                librosa.output.write_wav(
                    os.path.join(self.dest_dir, "wavs", audio), 
                    silence_removed, 
                    sr
                )
                """
                sf.write(
                    os.path.join(self.dest_dir, "wavs", audio),
                    silence_removed,
                    sr,
                    format='wav',
                    subtype='PCM_16'
                )

                filtered_audios.append(audio)
                filtered_txts.append(audio.rsplit('.', maxsplit=1)[0] + '.txt')

        self.len_shortest_audio = min(audio_lens)
        self.len_longest_audio = max(audio_lens)

        for text in filtered_txts:
            shutil.copyfile(
                os.path.join(self.concat_dir, text),
                os.path.join(self.dest_dir, "txts", text),
            )

        trimmed = []

        for wav, trans in zip(filtered_audios, filtered_txts):
            with open(os.path.join(self.concat_dir, trans)) as f:
                text = f.read().strip()
            trimmed.append([wav, text, text.lower()])

        trimmed = pd.DataFrame(trimmed, columns=["wav_file_name", "transcription", "transcription_normalized"])

        if not self.keep_audio_extension:
            trimmed["wav_file_name"] = trimmed["wav_file_name"].apply(
                lambda x: x.replace(WAV_SUFFIX, "")
            )
        trimmed['wav_file_name'] = trimmed['wav_file_name'].apply(lambda x: f'wavs/{x}')

        if self.output_type == "csv":
            trimmed["transcription"] = trimmed["transcription"].apply(
                lambda x: self.cleaner.clean_english_text(x)
            )

            trimmed.to_csv(
                os.path.join(self.dest_dir, "metadata.csv"),
                sep="|",
                index=None,
                header=None,
            )
            # Make validation & training datasets (lazy)
            DATA_SPLIT_FACTOR = 0.9
            trimmed[0:int(len(trimmed) * DATA_SPLIT_FACTOR)].to_csv(
                os.path.join(self.dest_dir, "train.csv"),
                sep="|",
                index=None,
                header=None,
            )
            trimmed[int(len(trimmed) * DATA_SPLIT_FACTOR):].to_csv(
                os.path.join(self.dest_dir, "val.csv"),
                sep="|",
                index=None,
                header=None,
            )
            tqdm.write(
                f"Dataset '{self.name}' has been generated. Wav files are placed in '{self.dest_dir}/wavs'. Transcription files are placed in '{self.dest_dir}/txts'."
            )
            tqdm.write(f"Metadata is placed in '{self.dest_dir}' as 'metadata.csv'.")
        elif self.output_type == "json":
            data = {}
            for ix in range(trimmed.shape[0]):
                name = trimmed.iloc[ix][0]
                text = trimmed.iloc[ix][1]
                data[name] = text
            with open(os.path.join(self.dest_dir, "alignment.json"), "w") as f:
                json.dump(data, f)
            tqdm.write(
                f"Dataset '{self.name}' has been generated. Wav files are placed in '{self.dest_dir}/wavs'. Transcription files are placed in '{self.dest_dir}/txts'."
            )
            tqdm.write(f"Metadata is placed in '{self.dest_dir}' as 'alignment.json'.")

        self.get_total_audio_length()

    def prepare_dataset(
            self,
            links_txt,
            sr=22050,
            download_youtube_data=True,
            max_concat_limit=30,
            concat_count=2,
            min_audio_length=3,
            max_audio_length=30,
    ):
        """
        A wrapper method for:
          download
          split_audios
          concat_audios
          finalize_dataset

        Downloads YouTube Videos as wav files(optional),
        splits the audios into chunks, joins the
        junks into reasonable audios and trims silence
        from the audios. Creates a metadata file as csv/json
        after the dataset has been generated.

        Parameters:
              links_txt: A .txt file that contains list of
                        video urls separated by new line.

              download_youtube_data:

              min_audio_length: The minimum length of audio files.

              max_audio_length: The maximum length of audio files.
        """
        self.sr = sr
        if download_youtube_data:
            # check if youtube file is already downloaded
            self.download(links_txt)
        self.split_audios()
        self.concat_audios(max_concat_limit, concat_count)
        self.finalize_dataset(min_audio_length, max_audio_length)

    def zip_dataset(self, zip_file, dataset_dir=None):
        """
        Zips the dataset into a zip file.
        """
        dataset_dir = dataset_dir or os.path.join(self.root, "datasets")

        shutil.make_archive(zip_file, "zip", self.dest_dir)

        tqdm.write(f"Dataset '{self.name}' has been zipped as '{zip_file}.zip'.")

