# Evaluating Perception Language Model (PLM)

[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20PLM&#160;1B-Model-blue)](https://huggingface.co/facebook/Perception-LM-1B)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20PLM&#160;3B-Model-blue)](https://huggingface.co/facebook/Perception-LM-3B)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20PLM&#160;8B-Model-blue)](https://huggingface.co/facebook/Perception-LM-8B) 

We have added our model and benchmarks to [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/models/simple/plm.py) for to support the process of reproducing our reported results on multiple image and video benchmarks.

---

## Getting Started
1. Install perception_models following the instruction in the [`Main README`](../../../README.md).
2. Install `lmms-eval`:
```
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmm-evals
pip install -e .
```

## Run Evaluation on Standard Image and Video Tasks
You can use the following command to run the evaluation.

```shell

# Use facebook/Perception-LM-1B for 1B parameters model and facebook/Perception-LM-8B for 8B parameters model.
CHECKPOINTS_PATH=facebook/Perception-LM-3B

# Define the tasks you want to evaluate PLM on. We support all the tasks present in lmms-eval, however have tested the following tasks with our models.

ALL_TASKS=(
    "docvqa" "chartqa" "textvqa" "infovqa" "ai2d_no_mask" "ok_vqa" "vizwiz_vqa" "mme"
    "realworldqa" "pope" "mmmu" "ocrbench" "coco_karpathy_val" "nocaps" "vqav2_val"
    "mvbench" "videomme" "vatex_test" "egoschema" "egoschema_subset" "mlvu_dev"
    "tempcompass_multi_choice" "perceptiontest_val_mc" "perceptiontest_test_mc"
)

# After specifying the task/tasks to evaluate, run the following command to start the evaluation.
SELECTED_TASK="textvqa,videomme"
accelerate launch --num_processes=8 \
-m lmms_eval \
--model plm \
--model_args pretrained=$CHECKPOINTS_PATH \
--tasks $SELECTED_TASK \
--batch_size 1 \
--log_samples \
--log_samples_suffix plm \
--output_path $OUTPUT_PATH
```
