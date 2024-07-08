import pickle


def save_tokenizer(tokenizer, target):
    with open(target, 'wb') as f:
        pickle.dump(tokenizer, f)


def load_tokenizer(pkl_file):
    with open(pkl_file, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer
