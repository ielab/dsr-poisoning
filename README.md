# dsr-poisoning: Document Screenshot Retrievers Poisoning
This is the official code repository for paper [Document Screenshot Retrievers are Vulnerable to Pixel Poisoning Attacks](https://arxiv.org/pdf/2501.16902), Shengyao Zhuang, Ekaterina Khramtsova, Xueguang Ma, Bevan Koopman, Jimmy Lin and Guido Zuccon.

## Installation

```bash
conda create -n dsr-poisoning python=3.10
conda activate dsr-poisoning

pip install torch torchvision
pip install vidore-benchmark
pip install transformers==4.48.2
pip install qwen-vl-utils
```

## Train adversarial document screenshot

We provided our data (target queries, seed images) used in the paper in the [data](./data) directory. Checkout the README in the directory for more details.

In this example we use [Trump's wikipedia page](trump_wiki.png) as the seed image and the target queries are [Wiki-SS-NQ](https://huggingface.co/datasets/Tevatron/wiki-ss-nq) training queries.
That is, optimize the Trump's wikipedia page to be retrieved by all training queries.

```bash
model=dse # or colpali
CUDA_VISIBLE_DEVICES=0 python dsr_attacks.py \
    --model $model \
    --seed_image ./trump_wiki.png \
    --target_query_file ./data/nq_train_queries/queries_kmeans_dse_1.json \
    --save_dir ./trump_$model_adv_images \
    --num_steps 3000 \
    --optimization_method direct
```
Here, the screenshot retriever is DSE model, change `--model dse` to `--model colpali` for ColPali retrieval model.

This attack config is **Direct Optimisation** with 100% gradient update.

For **Direct Optimisation** with less gradient update, for example 50%, set:
```bash
--optimization_method direct 
--grad_ratio 0.5
```

For **Noise Optimisation** with less gradient update, for example 50%, set: 
```bash
--optimization_method noise
--grad_ratio 0.5
```

For **Direct Mask Optimisation** with a mask margin ratio, for example 20%, set: 
```bash
--optimization_method direct 
--mask_ratio 0.2
```

## Evaluation
### Evaluate on [Wiki-SS-Corpus](Tevatron/wiki-ss-corpus) dataset

#### DSE eval
Following [tevatron DSE example](https://github.com/texttron/tevatron/tree/main/examples/dse/qwen) to encode wiki-ss-corpus to get query and document screenshot embeddings.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_nq_dse.py \
  --adv_image_dir ./trump_dse_adv_images \
  --corpus_embeddings_dir path_to_embedding_dir
  
 # {'success_1': 0.11966759002770083, 'success_5': 0.32797783933518004, 'success_10': 0.42077562326869805, 'success_100': 0.7030470914127424, 'mrr_100': 0.21772750989553608}
```

#### ColPali eval
Following [tevatron ColPali example](https://github.com/texttron/tevatron/tree/main/examples/colpali) to encode wiki-ss-corpus to get query and document screenshot embeddings.
Note, for ColPali we need to run file from tevatron search results.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_nq_colpali.py \
  --adv_image_dir ./trump_colpali_adv_images \
  --corpus_embeddings_dir path_to_embedding_dir \
  --run_file path_to_run_file
  
 # {'success_1': 0.09362880886426593, 'success_5': 0.19667590027700832, 'success_10': 0.25096952908587256, 'success_100': 0.521606648199446, 'mrr_100': 0.15225463089595362}
```

### Evaluate on VIDORE datasets
Since VIDORE datasets are small, we can encode the datasets for both DSE and ColPali models on-the-fly.
Set `--model` to `dse` or `colpali` to evaluate on DSE or ColPali model.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_vidore.py \
  --model dse \
  --adv_image_dir ./trump_dse_adv_images
```
