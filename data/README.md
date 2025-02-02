# Data
## nq_train_queries

The queries in this directory are k-means clustered queries from the [Wiki-SS-NQ](https://huggingface.co/datasets/Tevatron/wiki-ss-nq) training queries, with DSE embeddings.
`cluster_nq_train.py` is the script that we used to cluster the training queries. We provided our clustered queries in the following files:

`queries_kmeans_dse_1.json`: all the training queries that as a single cluster.

`queries_kmeans_dse_10.json`: training queries are clustered into 10 clusters.

`queries_kmeans_dse_100.json`: training queries are clustered into 100 clusters.


## seed_queries
The files in this directory contain the seed target query groups used in the paper. We sampled one query from each group in the Wiki-SS-NQ test queries as the seed query and used ChatGPT to generate nine similar queries.