# Perception Models: An Easy to Use Repository for Perception Tasks

**[Meta AI Research, FAIR](https://ai.facebook.com/research)**

---
Perception Models is a user-friendly repository designed to support the training, inference, and evaluation of Perception Language Model (PLM) and Perception Encoder (PE). It is designed to be modular and easy to expand and experiment with.

---

* **[Apr-17-25]:** Perception Encoder (PE) and Perception Language Model (PLM) are released. [[`Blog`](https://ai.meta.com/blog/meta-fair-updates-perception-localization-reasoning)] :fire::fire:

---

## Perception Encoder (PE)

[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-blue)](https://huggingface.co/collections/facebook/perception-encoder-67f977c9a65ca5895a7f6ba1) [![Paper](https://img.shields.io/badge/Technical%20Report-Perception%20Encoder-b31b1b.svg)](https://ai.meta.com/research/publications/perception-encoder-the-best-visual-embeddings-are-not-at-the-output-of-the-network)
[![Paper](https://img.shields.io/badge/arXiv-2504.13181-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2504.13181)
[![Model License](https://img.shields.io/badge/Model_License-Apache_2.0-olive)](https://opensource.org/licenses/Apache-2.0)

Perception Encoder (PE) is an extremely powerful and versatile family of vision encoders for both images and video: PE core can outperform SigLIP2 on Image CLIP and InternVideo2 on Video CLIP; PE lang can be used to outperform QwenVL2.5 and InternVL3 on vision language modeling; and PE spatial can outperform DINOv2 on dense prediction tasks. And all of this follows the same, easily scalable contrastive pretraining.

See [`apps/pe/README.md`](apps/pe/README.md) for more information and how to get started using them!



---

## Perception Language Model (PLM)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-blue)](https://huggingface.co/collections/facebook/perception-lm-67f9783f171948c383ee7498) [![Paper](https://img.shields.io/badge/Technical%20Report-PerceptionLM-b31b1b.svg)](https://ai.meta.com/research/publications/perceptionlm-open-access-data-and-models-for-detailed-visual-understanding)
[![Paper](https://img.shields.io/badge/arXiv-2504.13180-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2504.13180)
[![Model License](https://img.shields.io/badge/Model_License-FAIR_Research_License-lightgrey)](LICENSE.PLM)

PerceptionLM (PLM) is a family of open and fully reproducible models to facilitate research in vision-language modeling (VLM). In conjunction with PE, it is powerful enough to compete with the latest state-of-the-art VLMs such as InternVL3 and QwenVL2.5, while using _fully open data_. We also release the largest spatiotemporally annotated video dense captioning and fine-grained human activity recognition datasets to ever exist.

See [`apps/plm/README.md`](apps/plm/README.md) for details and how to get started!

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

---

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
