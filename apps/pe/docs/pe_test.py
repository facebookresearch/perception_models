import os, sys
import json
import argparse

import torch
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer, AutoModel

sys.path.append('../../../')
sys.path.append('../')
# import decord

if torch.cuda.is_available():
    print('GPU is available. Use GPU for this demo')
else:
    print('Use CPU for this demo')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
from clip_benchmark.datasets.builder import build_wds_dataset
from clip_benchmark.metrics import zeroshot_classification


AVAILABLE_PE_MODELS = ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224']
AVAILABLE_SIGLIP_MODELS = [
    # SigLIP2
    "google/siglip2-base-patch16-224",
    "google/siglip2-base-patch16-256",
    "google/siglip2-base-patch32-256",
    "google/siglip2-base-patch16-384",
    "google/siglip2-base-patch16-512",
    "google/siglip2-large-patch16-256",
    "google/siglip2-large-patch16-384",
    "google/siglip2-large-patch16-512",
    "google/siglip2-so400m-patch14-224",
    "google/siglip2-so400m-patch14-384",
    "google/siglip2-so400m-patch16-256",
    "google/siglip2-so400m-patch16-384",
    "google/siglip2-so400m-patch16-512",
    "google/siglip2-giant-opt-patch16-256",
    "google/siglip2-giant-opt-patch16-384",
    # SigLIP
    "google/siglip-base-patch16-224",
    "google/siglip-base-patch16-256",
    "google/siglip-base-patch16-384",
    "google/siglip-base-patch16-512",
    "google/siglip-large-patch16-256",
    "google/siglip-large-patch16-384",
    "google/siglip-so400m-patch14-224",
    "google/siglip-so400m-patch14-384",
]

WDS_DATASETS = ['wds_cars', 'wds_cifar10', 'wds_country211', 'wds_dollar_street', 'wds_fairface', 'wds_fgvc_aircraft', 'wds_food101', 'wds_geode', 'wds_gtsrb', 'wds_imagenet-a', 'wds_imagenet-o', 'wds_imagenet-r', 'wds_imagenet_sketch', 'wds_imagenetv2', 'wds_inaturalist', 'wds_mnist', 'wds_objectnet', 'wds_renderedsst2', 'wds_stl10', 'wds_sun397', 'wds_utkface', 'wds_voc2007', 'wds_vtab-caltech101', 'wds_vtab-cifar100', 'wds_vtab-clevr_closest_object_distance', 'wds_vtab-clevr_count_all', 'wds_vtab-dtd', 'wds_vtab-eurosat', 'wds_vtab-flowers', 'wds_vtab-kitti_closest_vehicle_distance', 'wds_vtab-pcam', 'wds_vtab-pets', 'wds_vtab-resisc45', 'wds_vtab-svhn', 'wds_wilds-camelyon17', 'wds_wilds-fmow', 'wds_wilds-iwildcam']


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, help="model name.")
    # parser.add_argument("--dataset_name", type=str, default="imagenet1k", help="Dataset specifier. See data.py.")
    parser.add_argument("--bs", type=int, default=256, help="Eval batch size.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloder workers.")
    args = parser.parse_args(args)
    return args


args = parse_args(sys.argv[1:])

# Load <LANG>_classnames.json (packaged with CLIP benchmark that are used by default)
default_classname_file = os.path.join(
    '/home/pengchuanzhang/GitHub/perception_models/apps/pe/clip_benchmark/datasets',  "en_classnames.json"
)
if os.path.exists(default_classname_file):
    with open(default_classname_file, "r") as f:
        default_classnames = json.load(f)
else:
    default_classnames = None

# Load <LANG>_zeroshot_classification_templates.json  (packaged with CLIP benchmark that are used by default)
default_template_file = os.path.join(
    '/home/pengchuanzhang/GitHub/perception_models/apps/pe/clip_benchmark/datasets',  "en_zeroshot_classification_templates.json"
)
if os.path.exists(default_template_file):
    with open(default_template_file, "r") as f:
        default_templates = json.load(f)
else:
    default_templates = None


model_name = args.model_name
if model_name in AVAILABLE_PE_MODELS:
    model = pe.CLIP.from_config(model_name, pretrained=True)  # Downloads from HF
    model = model.to(device)

    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
elif model_name in AVAILABLE_SIGLIP_MODELS:
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)

    image_size = int(model_name.split('-')[-1])
    preprocess = transforms.get_image_transform(image_size, to_RGB=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
else:
    raise ValueError(f"Not supported model: {model_name}!")


batch_size = args.bs
num_workers = args.workers
to_evaluate_datasets = ['wds_fgvc-aircraft', 'wds_cub-200', 'imagenet1k']
# to_evaluate_datasets = WDS_DATASETS
for dataset_name in to_evaluate_datasets:
    print(f"Run inference on {dataset_name}...")
    if dataset_name.startswith("wds"):
        if dataset_name in WDS_DATASETS:
            data_root = f'/fsx-onevision/pengchuanzhang/datasets/pe_datasets/wds/{dataset_name}_test'
        elif dataset_name == 'wds_fgvc-aircraft':
            data_root = '/fsx-onevision/pengchuanzhang/datasets/pe_datasets/wds/wds_fgvc-aircraft-sam3_test'
        elif dataset_name == 'wds_cub-200':
            data_root = '/fsx-onevision/pengchuanzhang/datasets/pe_datasets/wds/wds_cub-200-sam3_test'
        else:
            raise ValueError(f"Dataset {dataset_name} not supported yet!")
        dataset = build_wds_dataset(
            dataset_name, preprocess, split="test", data_dir=data_root
        )
        dataloader = torch.utils.data.DataLoader(
            dataset.batched(batch_size),
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
        )
        zeroshot_templates = (
            dataset.templates if hasattr(dataset, "templates") else None
        )
        classnames = dataset.classes if hasattr(dataset, "classes") else None
        assert (
            zeroshot_templates is not None and classnames is not None
        ), "Dataset does not support classification"
    elif dataset_name == 'imagenet1k':
        data_root = '/fsx-onevision/shared/data/imagenet_full_size'
        dataset = ImageFolder(
            root=os.path.join(data_root, "val"), transform=preprocess, 
        )
        dataset.classes = default_classnames["imagenet1k"]
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        zeroshot_templates = default_templates["imagenet1k"]
        classnames = default_classnames["imagenet1k"]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet!")


    classifier = zeroshot_classification.zero_shot_classifier(
        model, tokenizer, classnames, zeroshot_templates, device, amp=True
    )

    logits, target = zeroshot_classification.run_classification(
        model,
        classifier,
        dataloader,
        device,
        amp=True,
    )

    pred = logits.argmax(axis=1)
    (acc1,) = zeroshot_classification.accuracy(logits, target, topk=(1,))
    print("Top1 accuracy: ", acc1)

    output_root = f'/fsx-onevision/pengchuanzhang/output/pe_evals/{model_name}'
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    save_file = os.path.join(output_root, f"{dataset_name}.pt")
    torch.save(
        {
            "classifier": classifier, 
            "logits": logits,
            "target": target,
            "acc1": acc1
        },
        save_file
    )


    # metrics = zeroshot_classification.evaluate(
    #     model,
    #     dataloader,
    #     tokenizer,
    #     classnames,
    #     zeroshot_templates,
    #     device=device,
    # )
    # print(metrics)
