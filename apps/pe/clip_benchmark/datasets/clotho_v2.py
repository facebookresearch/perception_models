from torch.utils.data import Dataset
from subprocess import check_call
import os
import pandas
import shutil
import torch

def cache_file(url, outfile):
    if not os.path.exists(outfile):
        print("Downloading Clotho V2 dataset...")
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        check_call(["curl", "--url", url, "--output", outfile + ".tmp"])
        os.rename(outfile + ".tmp", outfile)


class ClothoV2(Dataset):
    def __init__(self, transform, location=os.path.expanduser("~/.cache/perception_encoder/clotho_v2")):
        self.df = self.setup_dataset(location)
        self.transform = transform

    def setup_dataset(self, location):
        url = "https://zenodo.org/records/4783391/files/clotho_audio_evaluation.7z?download=1"
        compressed_file = os.path.join(location, "clotho_audio_evaluation.7z")
        cache_file(url, compressed_file)

        extracted_dir = os.path.join(location, "extracted")
        if not os.path.exists(extracted_dir):
            if os.path.exists(extracted_dir + ".tmp"):
                shutil.rmtree(extracted_dir + ".tmp")
            assert shutil.which("7z"), "Please install 7zip to extract the Clotho V2 dataset (`conda install -c conda-forge p7zip`)"
            check_call(["7z", "x", compressed_file, f"-o{extracted_dir}.tmp" ])
            os.rename(extracted_dir + ".tmp", extracted_dir)

        url = "https://zenodo.org/records/4783391/files/clotho_captions_evaluation.csv?download=1"
        metadata_file = os.path.join(location, "clotho_captions_evaluation.csv")
        cache_file(url, metadata_file)
        df = pandas.read_csv(metadata_file)
        df["path"] = df["file_name"].apply(lambda x: os.path.join(extracted_dir, "evaluation", x))
        return df

    def __len__(self):
        return len(self.df)

    def collate_fn(self, batch):
        audios, captions = zip(*batch)
        return audios, captions

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        captions = [row[f"caption_{i}"] for i in range(1, 6)]
        return row["path"], captions
