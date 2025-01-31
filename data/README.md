# Data
## nq_train_queries

The queries in this directory are k-means clustered queries from the [Wiki-SS-NQ](https://huggingface.co/datasets/Tevatron/wiki-ss-nq) training queries, with DSE embeddings.
`cluster_nq_train.py` is the script that we used to cluster the training queries. We provided our clustered queries in the following files:

`queries_kmeans_dse_1.json`: all the training queries that as a single cluster.

`queries_kmeans_dse_10.json`: training queries are clustered into 10 clusters.

`queries_kmeans_dse_100.json`: training queries are clustered into 100 clusters.


## seed_queries
The files in this directory are the seed target query groups that we used in the paper. We sample one query for each group to be the seed query and use ChatGPT to generate the other 9 similar queries.
