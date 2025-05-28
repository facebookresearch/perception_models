model='hf-hub:timm/ViT-L-16-SigLIP2-384'
DATASETS=./clip_benchmark/tasks/wds_benchmarks.txt
#export PYTHONPATH=/home/berniehuang/git/perception_models/:$PYTHONPATH
export PYTHONPATH=/home/berniehuang/git/eval_0528/perception_models/:$PYTHONPATH

python -m clip_benchmark.cli eval \
    --skip_existing \
    --model $model \
    --dataset "$DATASETS" \
    --dataset_root "/checkpoint/vision_encoder/dataset01/benchmark/{dataset_cleaned}/" \
    --output "./benchmark_{pretrained}_{dataset}_{num_frames}_{model}_{language}_{task}.json" \
