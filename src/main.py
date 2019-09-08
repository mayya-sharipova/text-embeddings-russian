import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece
import json
import time

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


##### INDEXING #####

def index_data():
    print(f'Creating the {ES_INDEX_NAME} index.')
    client.indices.delete(index=ES_INDEX_NAME, ignore=[404])

    with open(INDEX_FILE) as index_file:
        source = index_file.read().strip()
        client.indices.create(index=ES_INDEX_NAME, body=source)

    docs = []
    count = 0

    with open(DATA_FILE) as data_file:
        for line in data_file:
            line = line.strip()
            doc = json.loads(line)
            docs.append(doc)
            count += 1

            if count % ES_BATCH_SIZE == 0:
                index_batch(docs)
                docs = []
                print("Indexed {} documents.".format(count))

        if docs:
            index_batch(docs)
            print("Indexed {} documents.".format(count))

    client.indices.refresh(index=ES_INDEX_NAME, request_timeout=1000)
    # merge to 1 segment
    client.indices.forcemerge(index = ES_INDEX_NAME, max_num_segments=1, request_timeout=1000)

    print("Done indexing.")

def index_batch(docs):
    titles = [doc["title"] for doc in docs]
    title_vectors = embed_text(titles)

    requests = []
    for i, doc in enumerate(docs):
        request = doc
        request["_op_type"] = "index"
        request["_index"] = ES_INDEX_NAME
        request["title_vector"] = title_vectors[i]
        requests.append(request)
    bulk(client, requests)

##### SEARCHING #####

def run_query_loop():
    while True:
        try:
            handle_query()
        except KeyboardInterrupt:
            return

def handle_query():
    query = input("Enter query: ")

    embedding_start = time.time()
    query_vector = embed_text([query])[0]
    embedding_time = time.time() - embedding_start

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc['title_vector']) + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    search_start = time.time()
    response = client.search(
        index=ES_INDEX_NAME,
        body={
            "size": ES_SEARCH_SIZE,
            "query": script_query,
            "_source": {"includes": ["title", "abstract"]}
        }
    )
    search_time = time.time() - search_start

    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    print("embedding time: {:.2f} ms".format(embedding_time * 1000))
    print("search time: {:.2f} ms".format(search_time * 1000))
    for hit in response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print("title: {}".format(hit["_source"]["title"]))
        print("abstract: {}".format(hit["_source"]["abstract"]))
        print()


def embed_text(text):
    vectors = session.run(embeddings, feed_dict={text_input: text})
    return [vector.tolist() for vector in vectors]



##### MAIN SCRIPT #####

if __name__ == '__main__':
    INDEX_FILE = "data/wikihow/index.json"
    DATA_FILE = "data/wikihow/russian_wikihow_title_abstract.json"

    ES_INDEX_NAME = "wikihow"
    ES_BATCH_SIZE = 500
    ES_SEARCH_SIZE = 10

    # Graph set up.
    g = tf.Graph()
    with g.as_default():
      text_input = tf.placeholder(dtype=tf.string, shape=[None])
      embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/1")
      embeddings = embed(text_input)
      init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()

    # Initialize session.
    session = tf.Session(graph=g)
    session.run(init_op)
    print("tensorflow session created")

    client = Elasticsearch()
    #index_data()
    run_query_loop()

    session.close()