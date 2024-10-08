import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from tqdm import tqdm
import random
random.seed(0)

# Generate key and query
def generate_key_query(num_keys, gold_key_min, gold_key_max):
    # Generate gold key, i.e., a random integer such that gold_key_min<gold_key<gold_key_max
    gold_key = random.randint(gold_key_min + 1, gold_key_max - 1)
    # Generate keys, integers not between gold_key_min and gold_key_max
    keys = random.sample(range(10000), num_keys * 10)
    keys = [key for key in keys if key < gold_key_min or key >= gold_key_max]
    keys = keys[:num_keys - 1]
    keys.append(gold_key)

    # Generate query
    query = "An integer smaller than " + str(gold_key_max) + " and larger than " + str(gold_key_min) + "."

    return keys, query, gold_key

if __name__ == '__main__':

    model_path = 'bge-m3'  # model name
    num_samples = 100  # number of samples
    num_keys = 20  # number of candidate keys
    max_value = 30  # maximum value of all the candidate keys

    query_prefix = "query: " if 'e5' in model_path else ""
    passage_prefix = "passage: " if 'e5' in model_path else ""
    prompts = {"query": query_prefix, "passage": passage_prefix}
    model = SentenceTransformer(model_path, device='cuda', prompts=prompts, trust_remote_code=True)


    correct_num = 0
    for i in tqdm(range(num_samples)):

        # Randomly select an integer within 0 to max_value
        gold_value = random.randint(1, max_value)

        # Generate query
        query = "An integer equal to {}.".format(gold_value)
        query_embedding = model.encode([prompts['query'] + query], device="cuda")

        # Get all integers within 0-999 that are not equal to gold_value_bound
        other_values = [i for i in range(max_value) if i != gold_value]
        # Randomly select num_kvs-1 integers from other_values
        nums = [gold_value] + [random.choice(other_values) for _ in range(num_keys - 1)]

        # Generate keys
        keys = [prompts['passage'] + str(num) for num in nums]
        key_embeddings = model.encode(keys, device="cuda")

        # Calculate similarity
        cos_scores = util.pytorch_cos_sim(query_embedding, key_embeddings)
        cos_scores = cos_scores.cpu().numpy().flatten()
        top_results = cos_scores.argsort()[::-1]

        # If the first key is the most similar, consider it correct
        if top_results[0] == 0:
            correct_num += 1

    print("Accuracy rate: ", correct_num / num_samples)





