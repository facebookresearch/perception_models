from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BatchFeature
from core.audio_visual_encoder import PEAudioVisual
from collections import defaultdict


def get_pe_av_embeddings(inputs, model, transform, this_modality, other_modality, device):
    if this_modality == "text":
        model_inputs = transform(text=inputs).to(device)
        if other_modality == "video":
            return model.encode_video_text(**model_inputs)
        elif other_modality == "audio":
            return model.encode_audio_text(**model_inputs)
        else:
            assert other_modality == "audio_video"
            return model.encode_audio_video_text(**model_inputs)
    elif this_modality == "audio":
        model_inputs = transform(audio=inputs).to(device)
        return model.encode_audio(**model_inputs)
    elif this_modality == "video":
        model_inputs = transform(videos=inputs).to(device)
        return model.encode_video(**model_inputs)
    else:
        assert this_modality == "audio_video"
        audios, videos = zip(*inputs)
        model_inputs = transform(audio=audios, videos=videos).to(device)
        return model.encode_audio_video(**model_inputs)


def evaluate(
    model,
    dataloader,
    tokenizer,
    device,
    video_dataset=False,
    amp=True,
    recall_k_list=[1, 5, 10],
    args=None,
    audio_dataset=False,
    transform=None,
):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------

    model: torch.nn,Module
        CLIP-like model with `encode_(image|video|audio)` and `encode_text`

    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers

    device: cpu/cuda

    amp: whether to use automatic mixed precision

    recall_k_list: list of int
        recall@k k's to use

    Returns
    -------

    dict of retrieval metrics
    """
    # list of batch of modality1 embedding
    batch_modality1_emb_list = []
    # list of batch of modality2 embedding
    batch_modality2_emb_list = []
    # for each text, we collect the corresponding media index, as each media can have multiple corresponding texts
    modality2_media_index = []
    dataloader_wrapper = dataloader_with_indices(dataloader)
    autocast = torch.cuda.amp.autocast if amp else suppress

    modality1 = "audio" if audio_dataset else "video" if video_dataset else "image"
    modality2 = "video" if audio_dataset and video_dataset else "text"

    all_modality2 = []

    for batch_modality1, batch_modality2, inds in tqdm(dataloader_wrapper):
        all_modality2.extend(batch_modality2)
        # store the index of media for each text
        if isinstance(inds, torch.Tensor):
            inds = inds.tolist()
        batch_modality2_media_index = [
            ind for ind, texts in zip(inds, batch_modality2) for text in texts
        ]
        # compute the embedding
        with torch.no_grad(), autocast():
            if isinstance(model, PEAudioVisual):
                batch_modality1_emb = get_pe_av_embeddings(
                    inputs=batch_modality1,
                    model=model,
                    transform=transform,
                    this_modality=modality1,
                    other_modality=modality2,
                    device=device,
                )
                batch_modality2_emb = get_pe_av_embeddings(
                    inputs=[i for batch in batch_modality2 for i in batch],
                    model=model,
                    transform=transform,
                    this_modality=modality2,
                    other_modality=modality1,
                    device=device,
                )
            else:
                # move the batch to the device
                if isinstance(batch_modality1, torch.Tensor):
                    batch_modality1 = batch_modality1.to(device, torch.float32)
                elif isinstance(batch_modality1, (list, tuple)):  # video frames as a list/tuple
                    batch_modality1 = [x.to(device, torch.float32) for x in batch_modality1]
                    batch_modality1 = torch.stack(batch_modality1, dim=0).permute(1,0,2,3,4).contiguous() # nbchw -> bncwh
                else:
                    raise NotImplementedError

                if video_dataset:
                    batch_modality1_emb = model.encode_video(batch_modality1, normalize=True)
                else:
                    batch_modality1_emb = model.encode_image(batch_modality1, normalize=True)
                tokenized = tokenizer(
                    [text for i, texts in enumerate(batch_modality2) for text in texts]
                ).to(device)
                batch_modality2_emb = model.encode_text(tokenized, normalize=True)

        batch_modality1_emb_list.append(batch_modality1_emb.cpu())
        batch_modality2_emb_list.append(batch_modality2_emb.cpu())
        modality2_media_index.extend(batch_modality2_media_index)

    batch_size = len(batch_modality1_emb_list[0])

    # concatenate all embeddings
    media_emb = torch.cat(batch_modality1_emb_list)
    texts_emb = torch.cat(batch_modality2_emb_list)

    # get the score for each text and media pair
    scores = texts_emb @ media_emb.t()

    scores_T = scores.T

    if args.reweight_retrieval:
        scores = scores * args.reweight_scale
        scores_T = scores_T * args.reweight_scale
        scores = scores * scores.softmax(dim=0)
        scores_T = scores_T * scores_T.softmax(dim=0)

    # construct a the positive pair matrix, which tells whether each text-media pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), modality2_media_index] = True

    all_modality2 = [x for y in all_modality2 for x in y]
    modality2_index_mapping = defaultdict(set)
    [modality2_index_mapping[modality2].add(i) for i, modality2 in enumerate(all_modality2)]
    for indices in modality2_index_mapping.values():
        if len(indices) > 1:
            # We have duplicate entries in modality2, so set them all to have the same labels
            index_list = list(indices)
            positive_pairs[index_list] = positive_pairs[index_list].any(dim=0)

    metrics = {}
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each media,  the number of retrieved texts matching that media among the top-k.
        # Also, the number of positives are the total number of texts matching the media in the dataset, as we have a set of captions
        # for each media, that number will be greater than 1 for text retrieval.
        # However, media/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each media, either 1 or 0. It is 1 if atleast one text matches the media among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each media (or text), we average
        # it over the dataset.
        metrics[f"{modality1}_retrieval_recall@{recall_k}"] = (
            (
                batchify(
                    recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k
                )
                > 0
            )
            .float()
            .mean()
            .item()
        )
        metrics[f"{modality2}_retrieval_recall@{recall_k}"] = (
            (
                batchify(
                    recall_at_k,
                    scores_T,
                    positive_pairs.T,
                    batch_size,
                    device,
                    k=recall_k,
                )
                > 0
            )
            .float()
            .mean()
            .item()
        )

    return metrics


def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(y)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and media embeddings (nb texts, nb media)
    :param k: number of media to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb media)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_media = scores.shape
    # for each text, sort according to media scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_media
    topk_indices_onehot = torch.nn.functional.one_hot(
        topk_indices, num_classes=nb_media
    )
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_media)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    # compute recall at k
    recall_at_k = nb_true_positive / nb_positive
    return recall_at_k


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)
