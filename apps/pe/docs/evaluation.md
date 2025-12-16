# Zero-Shot ClipBench Evaluation
Please download the supported datasets directly from the datasets host and update paths in clip_benchmark/datasets/builder.py. And run
```bash
model='PE-Core-G14-448'
DATASETS=./clip_benchmark/tasks/wds_benchmarks.txt
DATA_ROOT=DATA_ROOT/

python -m clip_benchmark.cli eval \
    --model $model \
    --pretrained $CHECKPOINT \
    --dataset "$DATASETS" \
    --dataset_root $DATA_ROOT \
    --output "./benchmark_{pretrained}_{dataset}_{num_frames}_{model}_{language}_{task}.json" \
    --force-preprocess-cfg resize_mode=squash

```
This script will perform zero-shot classification abd retireval benchmarks defined in clip_benchmark/tasks/wds_benchmarks.txt. Examples above includes the following tasks:
- ImageNet 1K classification
- ImageNet v2 classification
- ImageNet Adversial classification
- MS-COCO retrieval
- Flickr30K retrieval
- Kinetics 400 video classification
- MSR-VTT video retrieval



# Zero-Shot Retrieval for PE-AudioVisual

```bash
python -m clip_benchmark.cli eval \
    --model pe-av-large \
    --reweight-scale 10 \
    --dataset audiocaps-audio-video audiocaps-audio-text audiocaps-video-text clotho-v2 \
    --dataset_root $DATASETS \
    --output "./benchmark_{pretrained}_{dataset}_{num_frames}_{model}_{language}_{task}.json" \
    --batch_size 4 --no_amp
```

This will run zero-shot retrieval for the following tasks:
- Audiocaps Audio-Video
- Audiocaps Audio-Text
- Audiocaps Video-Text
- Clotho-V2 Audio-Text

Clotho-V2 will be downloaded from its original source and unpacked, but due to Audiocaps being a Youtube dataset, the user will need to provide the audio and video paths under `$DATASETS/audiocaps/audio` and `$DATASETS/audiocaps/video` respectively.
