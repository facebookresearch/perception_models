import torch
import os
from subprocess import check_call
import pandas
import warnings

class Audiocaps(torch.utils.data.Dataset):
    def __init__(self, transform, root):
        self.root = root
        self.transform = transform
        cache_dir = os.environ.get("PERCEPTION_MODELS_CACHE", os.path.expanduser("~/.cache/perception_encoder"))
        cache_dir = os.path.join(cache_dir, "audiocaps")
        os.makedirs(cache_dir, exist_ok=True)
        outfile = os.path.join(cache_dir, "test.csv")
        if not os.path.exists(outfile):
            url = "https://raw.githubusercontent.com/cdjkim/audiocaps/refs/heads/master/dataset/test.csv"
            check_call(["curl", url, "--output", outfile + ".tmp"])
            os.rename(outfile + ".tmp", outfile)
        self.df = pandas.read_csv(outfile)
        self.df = self.df.groupby(["youtube_id", "start_time"])["caption"].agg(list).reset_index()

        audio_files, video_files = [], []

        for yt_id in self.df["youtube_id"].values:
            audio_file = os.path.join(self.root, "audiocaps/audio", f"{yt_id}.flac")
            # Since youtube videos aren't permanent, we allow for some to be missing
            if not os.path.exists(audio_file):
                warnings.warn(f"Audio file {audio_file} does not exist, skipping...")
                audio_file = None
            audio_files.append(audio_file)
            video_file = os.path.join(self.root, "audiocaps/video", f"{yt_id}.mp4")
            if not os.path.exists(video_file):
                warnings.warn(f"Video file {video_file} does not exist, skipping...")
                video_file = None
            video_files.append(video_file)

        self.df["video_file"] = video_files
        self.df["audio_file"] = audio_files
        self.df = self.df[self.df["audio_file"].notnull() & self.df["video_file"].notnull()]

    def collate_fn(self, batch):
        source, target = zip(*batch)
        return source, target

    def __len__(self):
        return len(self.df)


class AudiocapsAudioVideo(Audiocaps):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["audio_file"], [row["video_file"]]

class AudiocapsAudioText(Audiocaps):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["audio_file"], row["caption"]


class AudiocapsVideoText(Audiocaps):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["video_file"], row["caption"]
