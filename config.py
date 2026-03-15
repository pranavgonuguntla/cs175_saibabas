import os
import random
import numpy as np
import torch

# Reproducibility
SEED = 175
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Device selection (CUDA > Apple Silicon MPS > CPU)
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("using Nvidia Cuda GPU")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = "mps"
    print("using Apple Silicon")
else:
    DEVICE = "cpu"
    print("using CPU")

# Dataset
YELP_DATASET_NAME = "yelp_polarity"

# Preprocessing
MAX_CHARS = 2000
MAX_WORD_TOKENS = 256
MIN_WORD_TOKENS = 20
MAX_WORD_TOKENS_FILTER = 200

# Keywords
USE_RAKE = False
KEYWORD_TOP_K = 8
CACHE_DIR = "./cache"
KEYWORD_CACHE_PATH = os.path.join(CACHE_DIR, "keyword_cache.json")

# Prompting / CFG
CFG_DROP_PROB = 0.15
MAX_KEYWORDS = 6
INCLUDE_SENTIMENT = True
NULL_PROMPT = "<NULL>"
LABEL_TO_SENTIMENT = {0: "negative", 1: "positive"}

# Dataset sizes
MAX_TRAIN_EXAMPLES = 560000
MAX_TEST_EXAMPLES = 38000

# DataLoader
BATCH_SIZE = 8

# GPT-2 baseline
GPT2_MODEL_NAME = "gpt2"
MAX_NEW_TOKENS = 150
NUM_SAMPLES = 2
GENERATION_SEED = 42
FINETUNE_EPOCHS = 3
FINETUNE_BATCH_SIZE = 8
FINETUNE_LR = 5e-5
FINETUNE_MAX_LENGTH = 256
FINETUNE_OUTPUT_DIR = os.path.join(os.getcwd(), "gpt2_yelp_finetuned")
