import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY  = os.environ["MISTRAL_API_KEY"]

PROPOSER_MODEL   = "mistral-small-latest"
CHALLENGER_MODEL = "mistral-small-latest"
DEVIL_MODEL      = "mistral-small-latest"
JUDGE_MODEL      = "mistral-small-latest"

NUM_ROUNDS       = 1
MAX_TOKENS       = 300
JUDGE_MAX_TOKENS = 500

CHROMA_DB_PATH   = "./chroma_db"
COLLECTION_NAME  = "debate_memory"
TOP_K_MEMORIES   = 3
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
