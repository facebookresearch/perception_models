# Training Perception Language Model (PLM)

[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20PLM&#160;Synthetic-Image-blue)](https://huggingface.co/datasets/facebook/PLM-Image-Auto)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20PLM&#160;Synthetic-Video-blue)](https://huggingface.co/datasets/facebook/PLM-Video-Auto)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20PLM&#160;Human-Video-blue)](https://huggingface.co/datasets/facebook/PLM-Video-Human)

We provide instruction to train or finetune PLM on a custom dataset.

---

> [!TIP]
> We provide configurations to run [`warm-up`](../configs/warmup/), [`sft`](../configs/sft/) to facilitate reproducibility of PLM training.


## Data Format :open_file_folder:

We use support both image and video conversation datasets using `jsonl`. Each line of `jsonl` file should follow the following format,

### For Image Conversation Dataset
```json
  {
    "image": "<image path>",
    "conversations": [
      {
        "from": "human",
        "value": "human instruction"
      },
      {
        "from": "assistant",
        "value": "model response"
      }
    ]
  }
```

### For Video Conversation Dataset
```json
  {
    "video": "<video path>",
    "conversations": [
      {
        "from": "human",
        "value": " human instruction"
      },
      {
        "from": "assistant",
        "value": "model response"
      }
    ]
  }
```

Note that for images, we require the `image` key to be present in the `jsonl line`, while for videos we require the `video` key to be present in the `jsonl line`. The `conversations` key is common between the two types.

> [!TIP]
> The repo also support `text-only`, `multi-image`, `image-region`, `video-region-caption (RCap)`, `video-region-temporal-localization (RTLoc)` and `video-region-dense-captioning (RDCap)` tasks. Please download the provided [`dummy-datasets`](https://dl.fbaipublicfiles.com/plm/dummy_datasets.tar.gz) for an example of each dataset.


### Registration of New Dataset
Given the dataset `jsonl` file, we can register a new dataset by adding an entry in [`apps/plm/configs/datasets.yaml`](apps/plm/configs/datasets.yaml).

```shell
custom_dataset_name:
    annotation: path/to/the/jsonl/file.jsonl
    root_dir: path/to/the/image-or-video/root-dir
```
Please refer to [`apps/plm/configs/datasets.yaml`](apps/plm/configs/datasets.yaml) for already present dummy image, video and grounding datasets.

---

## Training / Finetuning PLM :train:
Training PLM involves creating a `.yaml` configuration file, defining all model and training related configurable parameters. Please refer to the provided [`sft`](../configs/sft/) for details.

> [!TIP]
> To run the following code, download the [`dummy-datasets`](https://dl.fbaipublicfiles.com/plm/dummy_datasets.tar.gz) and extract them to `apps/plm/dummy_datasets`.

Given a `.yaml` configuration file, please run the following command to launch the training on a single node with 8 GPUs.

```shell
torchrun --nproc-per-node 8 -m apps.plm.train config=apps/plm/configs/finetune/plm_3b.yaml
```

### Consolidate Checkpoints
In order to run inference / evaluation, please consolidate checkpoints using the following command,

```shell
python apps/plm/consolidate.py --ckpt <path to the saved checkpoints.>
```

### Run Inference / Evaluation
After consoldating the checkpoints, you can run inference using the following command,

```shell
python apps/plm/generate.py \
--ckpt facebook/Perception-LM-3B \
--media_type image \  # Replace with "video" for running inference on video
--media_path <path to image or video> \
--question <Question to be asked about the video.>
```

For evaluation, please refer to [`evaluation.md`](evaluation.md).

---

We also provide a script to launch a distributed multinode training on slurm. Please use the provided utility named `stool.py`.

```shell
python -m core.stool script=apps.plm.train config=apps/plm/configs/finetune/plm_8b.yaml qos=<QoS> nodes=<num_of_nodes>
```

---

We provide a step-by-step example for how to finetune PLM on a public dataset that elaborates on each of the steps above in detail. Please see [`finetune_example.md`](finetune_example.md). 
