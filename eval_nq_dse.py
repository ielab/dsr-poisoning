import pickle
import torch
import numpy as np
import os
from PIL import Image
import glob
from tqdm import tqdm
from argparse import ArgumentParser
from utils import store_results
from models import DSE
import json


def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


def get_all_query_embeddings(embedding_dir):
    query_files = glob.glob(f'{embedding_dir}/query.*.pkl')
    query_embeddings = []
    for file in query_files:
        query_embeddings.append(pickle_load(file)[0])
    query_embeddings = np.concatenate(query_embeddings)
    query_embeddings = torch.tensor(query_embeddings).to('cuda')
    return query_embeddings


def get_target_query_embedding(seed_query_path, model):
    with open(seed_query_path, 'r') as f:
        print(f"Evaluating using the following seed queries: {seed_query_path}")
        queries = json.load(f)
        queries = queries['queries']
    with torch.no_grad():
        query_embeddings = model.get_query_embeddings(queries).detach()
    query_embeddings = query_embeddings.to(model.model.device)
    return query_embeddings


def eval_nq(args):
    adv_image_dir = args.adv_image_dir
    corpus_embeddings_dir = args.corpus_embeddings_dir
    cache_dir = args.cache_dir
    seed_query_path = args.seed_query_path

    model = DSE(cache_dir=cache_dir).to('cuda').eval()

    adv_images = []
    for file in os.listdir(f'{adv_image_dir}'):
        if file.endswith('.png') or file.endswith('.jpg'):
            adv_images.append(Image.open(f'{adv_image_dir}/{file}'))

    print(f'Number of adversarial images: {len(adv_images)}')

    with torch.no_grad():
        adv_img_embeddings = model.get_doc_embeddings(adv_images).float()

    if seed_query_path is not None:
        query_embeddings = get_target_query_embedding(seed_query_path, model)
    else: # load tevatron query pickle file
        print(f"Evaluating using all queries in {corpus_embeddings_dir}")
        query_embeddings = get_all_query_embeddings(corpus_embeddings_dir)

    corpus_files = glob.glob(f'{corpus_embeddings_dir}/corpus.*.pkl')
    doc_embeddings = []
    lookups = []
    for file in corpus_files:
        emb, lookup = pickle_load(file)
        doc_embeddings.extend(emb)
        lookups.extend(lookup)

    size_of_corpus = len(doc_embeddings)
    doc_embeddings = np.array(doc_embeddings)
    doc_embeddings = torch.tensor(doc_embeddings).to('cuda')
    doc_embeddings = torch.cat([doc_embeddings, adv_img_embeddings])
    lookups.extend([f'adv_{i}' for i in range(len(adv_img_embeddings))])

    lookup_adv_indices = torch.tensor([i for i in range(size_of_corpus, size_of_corpus + len(adv_img_embeddings))])
    lookup_adv_indices = lookup_adv_indices.to('cuda')

    k_range = [1, 5, 10, 100]
    metrics = {f'success_{k}': 0 for k in k_range}

    first_positions = []
    rr_adversarial = []

    for query_embed in tqdm(query_embeddings):
        similarities = model.compute_similarity(query_embed.unsqueeze(0), doc_embeddings)
        sorted_similarities, sorted_doc_ids = torch.sort(similarities, descending=True)

        adv_positions = torch.nonzero(torch.isin(sorted_doc_ids[0], lookup_adv_indices)).squeeze()

        # If there is only one adversarial image =>return its position
        # Otherwise, return the position of the highest adversarial image in the ranking
        if len(lookup_adv_indices) == 1:
            first_position = adv_positions.item()
        else:
            first_position = adv_positions[0].item()

        for k in k_range:
            if first_position < k:
                metrics[f'success_{k}'] += 1

        if first_position > 100:
            rr_adversarial.append(0)
        else:
            rr_adversarial.append(1 / (first_position + 1))

        first_positions.append(first_position)

    for k in k_range:
        metrics[f'success_{k}'] /= len(query_embeddings)
    metrics["mrr_100"] = float(np.mean(rr_adversarial))
    # metrics['mean_first_position'] = float(np.mean(first_positions))
    # metrics["first_positions"] = first_positions
    print(metrics)
    store_results(metrics, adv_image_dir, dset_name="dse_nq")

    return metrics


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--adv_image_dir', type=str, required=True, help='Path to adversarial images')
    parser.add_argument('--seed_query_path', type=str, default=None, help='Path to seed queries')
    parser.add_argument('--corpus_embeddings_dir', type=str, required=True, help='Path to corpus embeddings')
    parser.add_argument('--cache_dir', type=str, default=None)

    args = parser.parse_args()
    eval_nq(args)