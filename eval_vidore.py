import copy
from datasets import load_dataset
# from dotenv import load_dotenv
import math
from typing import Any, Dict, List, Optional
from PIL import Image
import os
import torch
from datasets import Dataset
from tqdm import tqdm
from vidore_benchmark.utils.iter_utils import batched
from models import Model, ColPali, DSE
from argparse import ArgumentParser
from utils import store_results


def embed_queries_passages(
        vision_retriever: Model,
        ds: Dataset,
        batch_query: int,
        batch_passage: int,
        img_resize: bool = False
):
    seen_queries = set()
    queries = []
    for query in ds["query"]:
        if query is not None and query not in seen_queries:
            queries.append(query)
            seen_queries.add(query)

    if len(queries) == 0:
        raise ValueError("All queries are None")

    # Get the embeddings for the queries and passages
    emb_queries = vision_retriever.get_query_embeddings(queries, batch_size=batch_query)

    # NOTE: To prevent overloading the RAM for large datasets, we will load the passages (images)
    # that will be fed to the model in batches (this should be fine for queries as their memory footprint
    # is negligible. This optimization is about efficient data loading, and is not related to the model's
    # forward pass which is also batched.
    emb_passages: List[torch.Tensor] = []

    for passage_batch in tqdm(
            batched(ds, n=batch_passage),
            desc="Dataloader pre-batching",
            total=math.ceil(len(ds) / (batch_passage)),
    ):
        passages: List[Any] = [db['image'] for db in passage_batch]
        if img_resize:
            resized_passages = [passage.resize((680, 680)) for passage in passages]
            passages = resized_passages

        batch_emb_passages = vision_retriever.get_doc_embeddings(passages)
        emb_passages.extend(batch_emb_passages)

    query_number = len(queries)
    assert len(emb_queries) == query_number

    return emb_queries, emb_passages


def evaluate_dataset_adv(
        emb_queries: torch.Tensor,
        emb_passages: List[torch.Tensor],
        vision_retriever: Model,
        adv_images: Optional[List[Image.Image]] = None,
) -> Dict[str, Optional[float]]:
    """
    Evaluate the model on a given dataset using the MTEB metrics.

    NOTE: The dataset should contain the following columns:
    - query: the query text
    - image_filename: the filename of the image
    - image: the image (PIL.Image) if `use_visual_embedding` is True
    - text_description: the text description (i.e. the page caption or the text chunks) if
        `use_visual_embedding` is False
    """

    # Dataset: sanity check
    # passage_column_name = "image" if vision_retriever.use_visual_embedding else "text_description"
    # required_columns = ["query", passage_column_name, "image_filename"]
    #
    # if not all(col in ds.column_names for col in required_columns):
    #     raise ValueError(f"Dataset should contain the following columns: {required_columns}")

    # Remove `None` queries (i.e. pages for which no question was generated) and duplicates
    # queries = list(set(ds["query"]))
    # --> old buggy behavior - this differs from colpali-engine implementation where duplicates are NOT removed
    # for fairness with externally evaluated retrievers since bug, we maintain this behavior and remove duplicates
    # This slightly boosts scores on docvqa typically

    # print("These values should be the same")
    # print(f'len(emb_queries): {len(emb_queries)}')
    # print(f'len(emb_passages): {len(emb_passages)}')

    # emb_adv_passages = vision_retriever.get_doc_embeddings(adv_images)
    # This gives OOM; instead, generate the embeddings in batches
    emb_adv_passages = []

    step_size = 8
    for i in range(0, len(adv_images), step_size):
        batch_images = adv_images[i:i+step_size]
        emb_adv_passages.extend(vision_retriever.get_doc_embeddings(batch_images))
    emb_adv_passages = torch.stack(emb_adv_passages)

    extended_emb_passages = copy.copy(emb_passages)
    extended_emb_passages.extend(emb_adv_passages)

    # Get the similarity scores
    scores = vision_retriever.compute_similarity(emb_queries, extended_emb_passages)

    total_num_docs = len(extended_emb_passages)
    num_adv_docs = len(emb_adv_passages)
    doc_ids = [str(i) for i in range(total_num_docs - num_adv_docs)]
    adv_doc_ids = [f'adv_{i}' for i in range(num_adv_docs)]
    doc_ids.extend(adv_doc_ids)

    metrics = {'success_5': 0,
               'success_10': 0,
               'success_20': 0,
               'success_100': 0}
    for i in range(len(emb_queries)):
        similarities = scores[i]
        # sort docs
        sorted_similarities, sorted_doc_ids = torch.sort(similarities, descending=True)
        sorted_doc_ids = [doc_ids[idx] for idx in sorted_doc_ids]
        for k in [5, 10, 20, 100]:
            if 'adv' in ''.join(sorted_doc_ids[:k]):
                metrics[f'success_{k}'] += 1

    for k in [5, 10, 20, 100]:
        metrics[f'success_{k}'] /= len(emb_queries)

    return metrics



# load_dotenv(override=True)
dataset_names = [
    'vidore/arxivqa_test_subsampled',
    'vidore/docvqa_test_subsampled',
    'vidore/infovqa_test_subsampled',
    'vidore/tabfquad_test_subsampled',
    'vidore/tatdqa_test',
    'vidore/shiftproject_test',
    'vidore/syntheticDocQA_artificial_intelligence_test',
    'vidore/syntheticDocQA_energy_test',
    'vidore/syntheticDocQA_government_reports_test',
    'vidore/syntheticDocQA_healthcare_industry_test'
]


def main(args):
    """
    Example script for a Python usage of the Vidore Benchmark.
    """
    adv_image_dir = args.adv_image_dir
    model_name = args.model
    cache_dir = args.cache_dir

    if model_name == 'dse':
        retriever = DSE(cache_dir=cache_dir).to('cuda')
        batch_passage= 1
    elif model_name == 'colpali':
        retriever = ColPali(cache_dir=cache_dir).to('cuda')
        batch_passage = 8
    else:
        raise ValueError('Invalid model name')

    for dataset_name in dataset_names:
        dataset = load_dataset(dataset_name, split="test", cache_dir=cache_dir)
        with torch.no_grad():
            query_emb, passage_emb = embed_queries_passages(retriever, dataset, batch_query=8,
                                                            batch_passage=batch_passage)

            # iterate through the experiment_ids in the adv_image_dir
            adv_images = []
            for file in os.listdir(adv_image_dir):
                if file.endswith('.jpg') or file.endswith('.png'):
                    adv_images.append(Image.open(f'{adv_image_dir}/{file}'))
            print("Number of adv images: ", len(adv_images))

            metrics = evaluate_dataset_adv(query_emb, passage_emb, retriever, adv_images)
            res = {dataset_name: metrics}
            store_results(res, adv_image_dir, dset_name=f"{model_name}_vidore")
            print(dataset_name, metrics)
        print(f"Finished {dataset_name}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='dse', help='model name. Either dse or colpali')
    parser.add_argument('--cache_dir', type=str, default=None, help='Cache directory to store the model')
    parser.add_argument('--adv_image_dir', type=str, required=True, help='Path to dir containing adversarial images')
    args = parser.parse_args()
    main(args)