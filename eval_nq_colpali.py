import pickle
import torch
import numpy as np
import os
from PIL import Image
import glob
from tqdm import tqdm

import json
from argparse import ArgumentParser
from utils import store_results
from models import DSE, ColPali

def read_run_file(run_file):
    res = {}
    with open(run_file, 'r') as f:
        for line in f:
            qid, docid, score = line.strip().split()
            score = float(score)
            if qid not in res:
                res[qid] = []
            res[qid].append((docid, score))
    return res

def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return reps, lookup


def get_all_query_embeddings(embedding_dir):
    query_files = glob.glob(f'{embedding_dir}/query.*.pkl')
    query_embeddings, qids = pickle_load(query_files[0])
    query_embeddings = query_embeddings.to('cuda')
    return query_embeddings, qids


def get_target_query_embedding(seed_query_path, model):
    with open(seed_query_path, 'r') as f:
        print(f"Evaluating using the following seed queries: {seed_query_path}")
        queries = json.load(f)
        queries = queries['queries']
    with torch.no_grad():
        query_embeddings = model.get_query_embeddings(queries).detach()
    query_embeddings = query_embeddings.to(model.model.device)
    return query_embeddings


def eval_nq_colpali(args):
    run_file = args.run_file
    adv_image_dir = args.adv_image_dir
    corpus_embeddings_dir = args.corpus_embeddings_dir
    cache_dir = args.cache_dir
    seed_query_path = args.seed_query_path
    model = ColPali(cache_dir=cache_dir).to('cuda').eval()

    adv_images = []
    for file in os.listdir(f'{adv_image_dir}'):
        if file.endswith('.png'):
            adv_images.append(Image.open(f'{adv_image_dir}/{file}'))

    with torch.no_grad():
        adv_img_embeddings = model.get_doc_embeddings(adv_images) #.float()


    if seed_query_path is not None:
        query_embeddings = get_target_query_embedding(seed_query_path, model)
        qids = [str(i) for i in range(len(query_embeddings))]
        if len(query_embeddings) == 0:
            raise ValueError("No queries found for the target query")
    else:
        query_embeddings, qids = get_all_query_embeddings(corpus_embeddings_dir)
        print(f"Evaluating using all queries in {corpus_embeddings_dir}")
    run = read_run_file(run_file)

    k_range = [1, 5, 10, 100]
    metrics = {f'success_{k}': 0 for k in k_range}
    first_positions,rr_adversarial = [], []
    # make dytype of query embeddings to model type and device

    for query_embed, qid in tqdm(zip(query_embeddings, qids)):
        query_embed = query_embed.to(model.model.device, dtype=adv_img_embeddings.dtype)
        adv_img_embeddings = adv_img_embeddings.to(model.model.device)
        similarities = model.compute_similarity(query_embed.unsqueeze(0), adv_img_embeddings)
        orignal_ranking = run[qid]


        for i, score in enumerate(similarities[0]):
            orignal_ranking.append((f'adv_{i}', score))
        sorted_doc_ids = [docid for docid, score in sorted(orignal_ranking, key=lambda x: -x[1])]

        first_position = next((i for i, docid in enumerate(sorted_doc_ids) if docid.startswith('adv_')), None)

        for k in k_range:
            if first_position < k:
                metrics[f'success_{k}'] += 1
        first_positions.append(first_position)

        # Calculate MRR@100         
        if first_position > 100:
            rr_adversarial.append(0)
        else:
            rr_adversarial.append(1 / (first_position + 1))


    for k in k_range:
        metrics[f'success_{k}'] /= len(query_embeddings)

    metrics["mrr_100"] = float(np.mean(rr_adversarial))
    print(metrics)
    # metrics['mean_first_position'] = float(np.mean(first_positions))
    # metrics["first_positions"] = first_positions

    store_results(metrics, adv_image_dir, dset_name="colpali_nq")

    return metrics

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--adv_image_dir', type=str)
    parser.add_argument('--corpus_embeddings_dir', type=str, required=True, help='Path to corpus embeddings')
    parser.add_argument('--run_file', type=str, required=True, help='Path to run file')
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--seed_query_path', type=str, default=None)

    args = parser.parse_args()
    eval_nq_colpali(args)
