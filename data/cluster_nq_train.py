from datasets import load_dataset
from models import DSE
from tqdm import tqdm
import torch
from sklearn.cluster import KMeans

dataset = load_dataset("Tevatron/wiki-ss-nq")

batch_size = 32
k=100

# Load the model
model = DSE().to("cuda").eval()

query_embeddings = []

# Get the query embeddings
for i in tqdm(range(0, len(dataset["train"]), batch_size)):
    queries = dataset["train"]['query'][i : i + batch_size]
    with torch.no_grad():
        query_embeddings.append(model.get_query_embeddings(queries))


# do kmeans to query embeddings close to 10 clusters
query_embeddings = torch.cat(query_embeddings, dim=0)
query_embeddings = query_embeddings.to("cpu").float()
kmeans = KMeans(n_clusters=k, random_state=0).fit(query_embeddings)

kmeans.cluster_centers_

clustered_queries = {}
for query, label in zip(dataset["train"]['query'], kmeans.labels_):
    if str(label) not in clustered_queries:
        clustered_queries[str(label)] = []
    clustered_queries[str(label)].append(query)

# save the clustered queries
import json
with open(f'./nq_train_queries/queries_kmeans_dse_{k}.json', 'w') as f:
    json.dump(clustered_queries, f)