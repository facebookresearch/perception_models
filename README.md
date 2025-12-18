# Perception Models: Powerful Models for Image, Video, and Audio Perception
[![Code License](https://img.shields.io/badge/Code_License-Apache_2.0-olive)](https://opensource.org/licenses/Apache-2.0)

This repo is the home to the state-of-the-art for image and video _perception_: [**Perception Encoder (PE)**](https://arxiv.org/abs/2504.13181) for image, video, [audio](https://ai.meta.com/research/publications/pushing-the-frontier-of-audiovisual-perception-with-large-scale-multimodal-correspondence-learning/) encoding, and [**Perception Language Model (PLM)**](https://arxiv.org/abs/2504.13180) for decoding.

> [!TIP]
> Click to Navigate!
>
> [Perception Encoder and Perception Encoder Audio-Visual](#perception-encoder-pe)
>
> [Perception Language Model](#perception-language-model-plm)
>
> [Dataset Releases](#dataset-releases)

## Updates
* **[Dec-16-25]:** We have released the Perception Encoder Audio-Visual (PE-AV) and Perception Encoder Audio-Frame (PE-A-Frame) models: [[`Blog`](https://ai.meta.com/blog/sam-audio/)][[`paper`](https://ai.meta.com/research/publications/pushing-the-frontier-of-audiovisual-perception-with-large-scale-multimodal-correspondence-learning/)] :fire::fire:
* **[Jul-14-25]:** PerceptionLM is now available in [Hugging Face transformers](https://huggingface.co/docs/transformers/main/en/model_doc/perception_lm). :fire::fire:
* **[Jul-11-25]:** We have release 8 new checkpoints for [Perception Encoder](apps/pe/README.md): 2x small core models (T and S), 2x tiling-tuned lang models (G and L), and 4x smaller spatial models (L, B, S, T). Give them a try! :fire::fire::fire:
* **[May-28-25]:** Perception Encoder has been integrated into [timm](https://github.com/huggingface/pytorch-image-models)! :fire::fire:
* **[Apr-18-25]:** Perception Language Model (PLM) and PLM-VideoBench are added to lmms-eval. This makes it easy to reproduce PLM results and allows you to evaluate on the PLM-VideoBench. [[`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/638)] :fire::fire:
* **[Apr-17-25]:** Perception Encoder (PE) and Perception Language Model (PLM) are released. [[`Blog`](https://ai.meta.com/blog/meta-fair-updates-perception-localization-reasoning)] :fire::fire:


## Perception Encoder (PE)
[![Data](https://img.shields.io/badge/Download-PE%20Data-ffcc00.svg)](https://huggingface.co/datasets/facebook/PE-Video)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-blue)](https://huggingface.co/collections/facebook/perception-encoder-67f977c9a65ca5895a7f6ba1)
[![Paper](https://img.shields.io/badge/Technical%20Report-Perception%20Encoder-b31b1b.svg)](https://ai.meta.com/research/publications/perception-encoder-the-best-visual-embeddings-are-not-at-the-output-of-the-network)
[![Paper](https://img.shields.io/badge/Technical%20Report-Perception%20Encoder%20AV-b31b1b.svg)](https://ai.meta.com/research/publications/pushing-the-frontier-of-audiovisual-perception-with-large-scale-multimodal-correspondence-learning/)
[![Paper](https://img.shields.io/badge/arXiv-2504.13181-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2504.13181)
[![Colab Demo](https://img.shields.io/static/v1?label=Demo&message=Google%20Colab&logo=google&color=orange)](https://colab.research.google.com/github/facebookresearch/perception_models/blob/main/apps/pe/docs/pe_demo.ipynb)
[![Model License](https://img.shields.io/badge/Model_License-Apache_2.0-olive)](https://opensource.org/licenses/Apache-2.0)

[Perception Encoder (PE)](https://arxiv.org/abs/2504.13181) is a family of the state-of-the-art vision and audio encoders for encoding images, video, and audio: PE core outperforms SigLIP2 on image and InternVideo2 on video benchmarks; PE lang can be used to outperform QwenVL2.5 and InternVL3 on vision language modeling; and PE spatial outperforms DINOv2 on dense prediction tasks. And all of this follows the same, easily scalable contrastive pretraining. Please see [README](apps/pe/README.md) for more details.

<img src="apps/pe/docs/assets/teaser.png" style="width: 100%; margin: 0 auto; display: block;" />

### Models
PE has 4 types of checkpoints, each excelling in a different area of computer vision and audio understanding:
 - [PE core](#vision-language-benchmarks): a CLIP model excels in vision-language tasks such as zero-shot image and video classification and video retrieval.
 - [PE lang](#multimodal-llm-benchmarks): a LLM-aligned PE that powers [PLM](https://arxiv.org/abs/2504.13180) to compete at the forefront of multimodal LLM benchmarks.
 - [PE spatial](#vision-centric-benchmarks): a spatially tuned PE that outperforms best spatial models for vision-centric tasks such as detection, depth estimation, and tracking.
 - [PE audio-visual](#audio-visual-benchmarks): a CLIP Model that embeds audio, video, audio-video, and text into a joint embedding space.

#### Vision-Language Benchmarks
|    | Model | Checkpoint | IN-1k | IN-v2 | IN-A | ObjectNet | COCO-T2I | Kinetics-400 | VTT-T2V
|:--:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|    | **T/16** 384px | [PE-Core-T16-384](https://huggingface.co/facebook/PE-Core-T16-384) | 62.1 | 54.7 | 21.1 | 43.9 | 33.0 | 41.5 | 28.8 |
|    | **S/16** 384px | [PE-Core-S16-384](https://huggingface.co/facebook/PE-Core-S16-384) | 72.7 | 65.0 | 49.5 | 60.0 | 42.6 | 55.0 | 39.3 |
|    | **B/16** 224px | [PE-Core-B16-224](https://huggingface.co/facebook/PE-Core-B16-224) | 78.4 | 71.7 | 62.4 | 71.9 | 50.9 | 65.6 | 47.6 |
|    | **L/14** 336px | [PE-Core-L14-336](https://huggingface.co/facebook/PE-Core-L14-336) | 83.5 | 77.9 | 89.0 | 84.7 | 57.1 | 73.4 | 50.3 |
|    | **G/14** 448px | [PE-Core-G14-448](https://huggingface.co/facebook/PE-Core-G14-448) | 85.4 | 80.2 | 92.6 | 88.2 | 58.1 | 76.9 | 51.2 |

#### Multimodal LLM Benchmarks

🔬 Controlled Setting:
|    | Encoder | Checkpoint | Doc VQA (val) | InfoQA (val) | TextVQA | MVBench | PerceptionTest (val) | EgoSchema (val) |
|:--:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|    | **L/14** 448px | [PE-Lang-L14-448](https://huggingface.co/facebook/PE-Lang-L14-448) | 81.9 | 46.4 | 73.0 | 52.3 | 54.7 | 59.8 |
|    | **G/14** 448px | [PE-Lang-G14-448](https://huggingface.co/facebook/PE-Lang-G14-448) | 84.4 | 48.3 | 75.2 | 52.4 | 56.0 | 62.0 |


🔥 SotA Setting:
|    | Model | Encoder | Doc VQA (test) | InfoQA (test) | TextVQA | MVBench | PerceptionTest (test) | EgoSchema (test) |
|:--:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|    | PLM-3B | [PE-Lang-L14-448-Tiling](https://huggingface.co/facebook/PE-Lang-L14-448-Tiling)* | 93.8 | 74.6 | 84.3 | 74.7 | 79.3 | 66.9 |
|    | PLM-8B | [PE-Lang-G14-448-Tiling](https://huggingface.co/facebook/PE-Lang-G14-448-Tiling)* | 94.6 | 80.9 | 86.5 | 77.1 | 82.7 | 68.8 |

\* These checkpoints were aligned with tiling. Use them if you use higher than 448 resolution with tiling in the LLM decoder.

#### Vision-centric Benchmarks
🦾 Main model:
|    | Encoder | Checkpoint | ADE20k <br/> [Segmentation](https://github.com/open-mmlab/mmsegmentation)<br />Linear Probe mIoU | DAVIS<br /> [Tracking](https://github.com/facebookresearch/dino/blob/main/eval_video_segmentation.py) <br />Zero-Shot J&F  | LVIS <br /> [Mask R-CNN](../detection/detectron2_pe/) 1024px <br /> Box / Mask mAP | COCO <br/> [DETA](../detection/DETA_pe/) 1824px <br /> Box mAP |
|:--:|:---:|:---:|:---:|:---:|:---:|:---:|
|    | **G/14** 448px | [PE-Spatial-G14-448](https://huggingface.co/facebook/PE-Spatial-G14-448) | 49.3 | 61.5 | 54.2 / 49.3 | 66.0 |


<div align="center">
  <img src="apps/pe/docs/assets/spatial_correspondence.png" style="width: 80%; margin: 0 auto; padding-top: 20px; padding-bottom: 20px; display: block;" />

  Visualization of PCA of non-maked visual tokens, mapped to RGB values.
</div>

⚗️ Distilled Models:
|    | Encoder<br />(Distilled from G) | Checkpoint | ADE20k <br/> [Segmentation](https://github.com/open-mmlab/mmsegmentation)<br />Linear Probe mIoU | DAVIS<br /> [Tracking](https://github.com/facebookresearch/dino/blob/main/eval_video_segmentation.py) <br />Zero-Shot J&F  |
|:--:|:---:|:---:|:---:|:---:|
|    | **T/16** 512px | [PE-Spatial-T16-512](https://huggingface.co/facebook/PE-Spatial-T16-512) | 27.6 | 55.0 |
|    | **S/16** 512px | [PE-Spatial-S16-512](https://huggingface.co/facebook/PE-Spatial-S16-512) | 37.5 | 57.5 |
|    | **B/16** 512px | [PE-Spatial-B16-512](https://huggingface.co/facebook/PE-Spatial-B16-512) | 44.4 | 58.9 |
|    | **L/14** 448px | [PE-Spatial-L14-448](https://huggingface.co/facebook/PE-Spatial-L14-448) | 48.1 | 60.6 |

See paper for comparison to other models.

#### Audio-Visual Benchmarks

|    | Model | Checkpoint   | Avg Retrieval | AudioCaps T→A | AudioCaps T→V | AudioCaps V→A | Clotho T→A | Valor T→A | Valor T→V | VCTK A→T | VGGSound V→A | Internal V→A |
|:--:|:-----:|--------------|---------------|---------------|---------------|---------------|------------|-----------|-----------|----------|---------------|---------------|
| 🆕 | **AV S** 16 frames | [`pe-av-small-16-frame`](https://huggingface.co/facebook/pe-av-small-16-frame)  | 45.2          | 41.2          | 18.6          | 75.4          | 24.0       | 29.8      | 70.1      | 96.1     | 34.1          | 17.9          |
| 🆕 | **AV B** 16 frames | [`pe-av-base-16-frame`](https://huggingface.co/facebook/pe-av-base-16-frame)   | 47.0          | 43.1          | 19.8          | 80.6          | 23.4       | 31.9      | 70.0      | 94.8     | 39.0          | 20.4          |
| 🆕 | **AV L** 16 frames | [`pe-av-large-16-frame`](https://huggingface.co/facebook/pe-av-large-16-frame)  | 48.2          | 44.7          | 19.5          | 86.1          | 22.8       | 35.0      | 70.9      | 85.6     | 45.2          | 23.9          |
| 🆕 | **AV S** all frames | [`pe-av-small`](https://huggingface.co/facebook/pe-av-small)           | 48.1          | 41.8          | 18.8          | 77.4          | 23.9       | 29.3      | 70.9      | 94.9     | 35.4          | 40.5          |
| 🆕 | **AV B** all frames | [`pe-av-base`](https://huggingface.co/facebook/pe-av-base)            | 50.2          | 42.7          | 19.6          | 83.7          | 23.8       | 30.8      | 71.2      | 94.9     | 40.7          | 44.6          |
| 🆕 | **AV L** all frames | [`pe-av-large`](https://huggingface.co/facebook/pe-av-large)           | 51.6          | 45.8          | 20.8          | 88.3          | 23.0       | 35.1      | 70.9      | 85.6     | 48.3          | 46.5          |

#### Audio Event Localization Benchmarks

|    | Model |     Checkpoint   | Internal Bench (AUROC)      | ASFX-SED (AUROC)        | AudioSet-Strong (AUROC) | DESED (AUROC) | UrbanSED (AUROC) |
|:--:|:-----:|------------------|---------------------|------------------|-----------------------|-------------|-------------|
| 🆕 | **A-Frame S** | [`pe-a-frame-small`](https://huggingface.co/facebook/pe-a-frame-small)| 0.91                | 0.83             | 0.96                  | 0.96        | 0.88        |
| 🆕 | **A-Frame B** | [`pe-a-frame-base`](https://huggingface.co/facebook/pe-a-frame-base)| 0.92                | 0.83             | 0.96                  | 0.98        | 0.89        |
| 🆕 | **A-Frame L** | [`pe-a-frame-large`](https://huggingface.co/facebook/pe-a-frame-large)| 0.91                | 0.83             | 0.96                  | 0.97        | 0.89        |

### Getting Started with PE
You can get started with the following example for image and text feature extraction or use our [Colab Demo](https://colab.research.google.com/github/facebookresearch/perception_models/blob/main/apps/pe/docs/pe_demo.ipynb)

```python
import torch
from PIL import Image
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

print("CLIP configs:", pe.CLIP.available_configs())
# CLIP configs: ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224', 'PE-Core-S16-384', 'PE-Core-T16-384']

model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # Downloads from HF
model = model.cuda()

preprocess = transforms.get_image_transform(model.image_size)
tokenizer = transforms.get_text_tokenizer(model.context_length)

image = preprocess(Image.open("docs/assets/cat.png")).unsqueeze(0).cuda()
text = tokenizer(["a diagram", "a dog", "a cat"]).cuda()

with torch.no_grad(), torch.autocast("cuda"):
    image_features, text_features, logit_scale = model(image, text)
    text_probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[0.0, 0.0, 1.0]]
```

> [!TIP]
> See [`apps/pe/README.md`](apps/pe/README.md) for details and how to get started!

### Getting Started with PE-AV

```python
import os
from core.audio_visual_encoder import PEAudioVisual, PEAudioVisualTransform
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PEAudioVisual.from_config("pe-av-large", pretrained=True).to(device)
transform = PEAudioVisualTransform.from_config("pe-av-large")

video_files = ["assets/train.mp4", "assets/office.mp4"]
descriptions = [
    "A person talking with sirens and a train in the background",
    "Two people talking in an office, with sounds of workers typing on a keyboard"
]

def embed(videos=None, audio=None, text=None):
    inputs = transform(videos=videos, audio=audio, text=text)
    inputs = inputs.to(device)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        return model(**inputs)

vt_outputs = embed(videos=video_files, text=descriptions)
avt_outputs = embed(videos=video_files, audio=video_files, text=descriptions)
at_outputs = embed(audio=video_files, text=descriptions)

# Compute dot product between visual and text
vt_dot_products = torch.einsum("ij,ij->i", vt_outputs.visual_embeds, vt_outputs.visual_text_embeds)
# Compute dot product between audio_visual and text
avt_dot_products = torch.einsum("ij,ij->i", avt_outputs.audio_visual_embeds, avt_outputs.audio_visual_text_embeds)
# Compute dot product between audio and text
at_dot_products = torch.einsum("ij,ij->i", at_outputs.audio_embeds, at_outputs.audio_text_embeds)
# Compute dot product between audio and video
av_dot_products = torch.einsum("ij,ij->i", avt_outputs.audio_embeds, avt_outputs.video_embeds)
```

### Getting Started with PE-A-Frame

```python
from core.audio_visual_encoder import (
    PEAudioFrame,
    PEAudioFrameTransform,
)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PEAudioFrame.from_config("pe-a-frame-large", pretrained=True).to(device)
transform = PEAudioFrameTransform.from_config("pe-a-frame-large")

descriptions = ["a person talking"]
inputs = transform(
    audio=["assets/office.mp4"],
    text=descriptions,
).to(device)

with torch.inference_mode():
    outputs = model(**inputs)

# Print the spans for each description (start and end timestamps for when they occur in the audio)
for description, spans in zip(descriptions, outputs.spans):
    span_str = ", ".join([f"({start:.2f}, {end:.2f})" for start, end in spans])
    print(f'"{description}": [{span_str}]')

```

> [!TIP]
> See [`apps/pe/README.md`](apps/pe/README.md) for additional details!

## Perception Language Model (PLM)
[![Data](https://img.shields.io/badge/Download-PLM%20Data-ffcc00.svg)](https://huggingface.co/datasets/facebook/PLM-Video-Human)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-blue)](https://huggingface.co/collections/facebook/perception-lm-67f9783f171948c383ee7498)
[![Paper](https://img.shields.io/badge/Technical%20Report-PerceptionLM-b31b1b.svg)](https://ai.meta.com/research/publications/perceptionlm-open-access-data-and-models-for-detailed-visual-understanding)
[![Paper](https://img.shields.io/badge/arXiv-2504.13180-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2504.13180)
[![Colab](https://img.shields.io/badge/Google%20Colab-Tutorials-red)](apps/plm/notebook_demos)
[![ModelLicense](https://img.shields.io/badge/Model_License-FAIR_Research_License-lightgrey)](LICENSE.PLM)

PerceptionLM (PLM) is a family of open and fully reproducible models to facilitate research in vision-language modeling (VLM). In conjunction with PE, it is powerful enough to compete with the latest state-of-the-art VLMs such as InternVL3 and QwenVL2.5, while using _fully open data_. We also release the largest spatiotemporally annotated video dense captioning and fine-grained human activity recognition datasets to ever exist.

![Description of the image](apps/plm/docs/plm_main_fig.png)

### Models
PLM releases models in three different sizes (1B, 3B and 8B).
* [Perception-LM-1B](https://huggingface.co/facebook/Perception-LM-1B): A PLM model trained using Llama-3.2-1B-Instruct base LLM.
* [Perception-LM-3B](https://huggingface.co/facebook/Perception-LM-3B): A PLM model trained using Llama-3.2-3B-Instruct base LLM.
* [Perception-LM-8B](https://huggingface.co/facebook/Perception-LM-8B): A PLM model trained using Llama-3.1-8B-Instruct base LLM.

#### PLM Image Benchmark Results

| Model  | DocVQA | ChartQA | TextVQA | InfoQA | AI2D  | OCRBench | COCO | Nocap | Flickr | MMMU | VQAv2 | OKVQA | VizWiz | MME | SEED | BLINK | CVBench | RealWorldQA | VSR | POPE |
|:---------:|:--------:|:---------:|:---------:|:--------:|:------:|:----------:|:------------:|:-------------:|:--------------:|:------:|:-------:|:--------:|:--------:|:-----:|:------:|:-------:|:----------:|:-------------:|:-----:|:------:|
| PLM1B  | 90.7   | 78.6    | 82.1    | 63.0   | 84.9 | 807      | 138.6      | 124.2       | 100.5        | 34.8 | 81.7  | 61.0   | 59.7   | 1603| 76.3 | 46.8  | 73.8     | 67.1        | 68.8| 88.4 |
| PLM3B  | 93.8   | 84.3    | 84.3    | 74.6   | 90.9 | 830      | 144.9      | 126.5       | 98.0         | 41.2 | 84.3  | 66.8   | 64.0   | 1879| 78.5 | 55.4  | 81.4     | 72.4        | 80.4| 88.7 |
| PLM8B  | 94.6   | 85.5    | 86.5    | 80.9   | 92.7 | 870      | 146.7      | 129.9       | 105.6        | 46.1 | 85.6  | 69.6   | 67.0   | 1989| 79.3 | 56.0  | 81.3     | 75.0        | 82.8| 89.9 |

#### PLM Video Benchmark Results

| Model  | VATEX                    | DREAM&nbsp;1K      | How2QA       | MVBench      | NExTQA      | PerceptionTest&nbsp;(test) | STAR       | TVQA       | VideoMME        | TVBench      | ActivityNetQA   | EgoSchema&nbsp;(test) | TemporalBench    | TOMATO     | MotionBench&nbsp;(dev) | TempCompass&nbsp;(MCQ) | CGBench&nbsp;(clue) | Charades&nbsp;STA   | VideoHallucer   | Halluc.&nbsp;EventHallusion |
|:-------------:|:---------------------------:|:-----------------------:|:---------------------:|:-------------:|:-------------:|:--------------------------:|:----------:|:----------:|:----------------:|:-------------:|:--------------------:|:----------------------:|:---------------------:|:------------:|:------------------------:|:-----------------------:|:---------------------:|:-------------------:|:-------------------------------:|:--------------------------------:|
| PLM1B  | 92.5 | 34.3 | 86.4 | 70.1 | 80.3 | 72.7 | 83.7 | 50.3 | 49.2 | 50.4 | 62.5 | 60.4 | 18.2 | 25.5 | 52.2 | 64.6 | 43.6 | 55.2 | 49.2 | 79.5 |
| PLM3B  | 96.1 | 37.4 | 89.4 | 74.7 | 83.4 | 79.3 | 84.8 | 55.3 | 54.9 | 58.9 | 66.2 | 66.9 | 23.4 | 30.9 | 60.4 | 69.3 | 47.2 | 57.7 | 55.5 | 76.5 |
| PLM8B  | 99.7 | 35.9 | 90.7 | 77.1 | 84.1 | 82.7 | 84.9 | 59.3 | 58.3 | 63.5 | 67.3 | 68.8 | 28.3 | 33.2 | 61.4 | 72.7 | 46.4 | 58.6 | 57.7 | 77.3 |

### PLM Resources

| Resource | Description | Documentation                                          |
| --- | --- |--------------------------------------------------------|
| **Evaluation** | Evaluation of PLM using lmms-eval | [`docs/evaluation.md`](apps/plm/docs/evaluation.md)    |
| **Training / Finetuning** | Training and finetuning instructions for PLM | [`docs/training.md`](apps/plm/docs/training.md)                 |
| **PLM-VideoBench** | Evaluation on PLM-VideoBench using lmms-eval | [`docs/plm_videobench.md`](apps/plm/docs/plm_videobench.md)     |
| **End-to-End Finetuning Example** | End-to-end finetuning example on radiology images | [`docs/finetune_example.md`](apps/plm/docs/finetune_example.md) |
| **Generating Response** | Generate responses using a trained model with `generate.py` | [`generate.py`](apps/plm/generate.py)                           |


> [!TIP]
> See [`apps/plm/README.md`](apps/plm/README.md) for details and how to get started!

## Dataset Releases


### 🎥 [PE-Video-Dataset (PVD)](https://huggingface.co/datasets/facebook/PE-Video)


PVD comprises 1M high quality and diverse videos. Among them, 120K videos are accompanied by automated and human-verified annotations. and all videos are accompanied with video description and keywords. The videos are motion-centered, covering both first-person and third-person views with a wide coverage of scenes.

🔹 [**PVD**](https://huggingface.co/datasets/facebook/PE-Video) - 1M High-Quality Human Annotated Video Dataset

<table>
   <tr>
    <td colspan="2" align="center"><strong>PVD</strong></td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/ead8a7ed-4d5b-465a-a396-68948683dfcf" alt="output_2" width="300"/><br>
      A person's hands pruning a plant with green leaves.
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/9e509e49-f550-4c5c-9571-ed57c5118227" alt="output" width="300"/><br>
      A detailed diorama of a rural landscape featuring a horse-drawn carriage moving along a dirt path
    </td>
  </tr>
</table>

---


### 🎥 [PLM-Video-Human](https://huggingface.co/datasets/facebook/PLM-Video-Human)

PLM-Video-Human is a collection of human-annotated resources for training Vision Language Models, focused on detailed video understanding. Training tasks include:

🔹 [**FGQA**](https://huggingface.co/datasets/facebook/PLM-Video-Human#fine-grained-question-answering-fgqa) — Fine-Grained Question Answering
🔹 [**RTLoc**](https://huggingface.co/datasets/facebook/PLM-Video-Human#region-temporal-localization-rtloc) — Region-Temporal Localization
🔹 [**RCap**](https://huggingface.co/datasets/facebook/PLM-Video-Human#region-video-captioning-rcap) — Region Video Captioning
🔹 [**RDCap**](https://huggingface.co/datasets/facebook/PLM-Video-Human#region-dense-temporal-captioning-rdcap) — Region Dense Temporal Captioning

<table>
  <tr>
    <td colspan="2" align="center"><strong>FGQA</strong></td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="https://github.com/user-attachments/assets/4f5c6c5e-687d-49df-9bf8-db9ec7f1f281" alt="fgqa" width="500"/>
    </td>
  </tr>
  <tr>
    <th>Question</th>
    <th>Answer</th>
  </tr>
  <tr>
    <td>In what direction do you move the tool while removing the shell?</td>
    <td>Both clockwise and anticlockwise.</td>
  </tr>
</table>

<table>
   <tr>
    <td colspan="2" align="center"><strong>STC</strong></td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="https://github.com/user-attachments/assets/a2a129c7-c1e9-47b5-a3b4-fc96a237a9fb" alt="stc" width="500"/>
    </td>
  </tr>
  <tr>
    <th>Time (s) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>[0, 4]</td>
    <td>The masked subject is a young boy wearing a red jacket and gray pants. He is grasping a monkey bar–like activity in a playground.</td>
  </tr>
  <tr>
    <td>[5, 14]</td>
    <td>He lets go of his hands and runs to the right side of the frame.</td>
  </tr>
  <tr>
    <td>[15, 30]</td>
    <td>The subject is out of frame.</td>
  </tr>
  <tr>
    <td>[31, 45]</td>
    <td>The subject runs back into the frame toward the higher monkey bar in the playground.</td>
  </tr>
  <tr>
    <td>[46, 74]</td>
    <td>He jumps underneath the metal bar and looks up at it. A man wearing a white polo runs toward the subject.</td>
  </tr>
  <tr>
    <td>[75, 116]</td>
    <td>The man in the white polo lifts the subject upward so he can grasp the higher metal bar. The subject holds onto the bar and hangs from it.</td>
  </tr>
</table>

---

### 🤖 Auto-Generated Datasets

Sythetic image/video captions and QAs used in PLM, please refer to the paper, Section 3 (PLM), for more details. The sythetic annotations covers: SA1B, Openimages, Obejct365, ArxivQA, UCSF, PDFAcc, YT-1B, Ego4d with captions, YT-1B with MCQAs and Ego4d with QAs.

🖼️ [**PLM-Image-Auto**](https://huggingface.co/datasets/facebook/PLM-Image-Auto) — Automatically generated image datasets

📹 [**PLM-Video-Auto**](https://huggingface.co/datasets/facebook/PLM-Video-Auto) — Automatically generated video datasets


---

## Installation :wrench:
```shell
git clone https://github.com/facebookresearch/perception_models.git
cd perception_models

conda create --name perception_models python=3.12
conda activate perception_models

# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers --index-url https://download.pytorch.org/whl/cu124

# We use torchcodec for decoding videos into PyTorch tensors
conda install ffmpeg -c conda-forge
pip install torchcodec==0.1 --index-url=https://download.pytorch.org/whl/cu124

pip install -e .
```
This will install an editable version of repo, allowing you to make changes to the code without needing to reinstall the package every time.


## 🙏 Acknowledgement
We are thankful to [Meta Lingua](https://github.com/facebookresearch/lingua) for releasing their code as open-source contributions. The code structure and code implementation of the LLM is directly forked from [Meta Lingua](https://github.com/facebookresearch/lingua). We are also thankful to [Open_CLIP](https://github.com/mlfoundations/open_clip) for open-source contributions in CLIP training, and [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark) for CLIP model evaluation.


## 📜 Citation
```BibTeX
@article{bolya2025PerceptionEncoder,
  title={Perception Encoder: The best visual embeddings are not at the output of the network},
  author={Daniel Bolya and Po-Yao Huang and Peize Sun and Jang Hyun Cho and Andrea Madotto and Chen Wei and Tengyu Ma and Jiale Zhi and Jathushan Rajasegaran and Hanoona Rasheed and Junke Wang and Marco Monteiro and Hu Xu and Shiyu Dong and Nikhila Ravi and Daniel Li and Piotr Doll{\'a}r and Christoph Feichtenhofer},
  journal={arXiv:2504.13181},
  year={2025}
}

@article{cho2025PerceptionLM,
  title={PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding},
  author={Jang Hyun Cho and Andrea Madotto and Effrosyni Mavroudi and Triantafyllos Afouras and Tushar Nagarajan and Muhammad Maaz and Yale Song and Tengyu Ma and Shuming Hu and Hanoona Rasheed and Peize Sun and Po-Yao Huang and Daniel Bolya and Suyog Jain and Miguel Martin and Huiyu Wang and Nikhila Ravi and Shashank Jain and Temmy Stark and Shane Moon and Babak Damavandi and Vivian Lee and Andrew Westbury and Salman Khan and Philipp Kr\"{a}henb\"{u}hl and Piotr Doll{\'a}r and Lorenzo Torresani and Kristen Grauman and Christoph Feichtenhofer},
  journal={arXiv:2504.13180},
  year={2025}
}
```
