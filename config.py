from pathlib import Path
import os
from dotenv import load_dotenv

# -----------------------------
# Project Root
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv(BASE_DIR / ".env")

# -----------------------------
# API Key
# -----------------------------
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found. Please check your .env file.")

# -----------------------------
# Paths
# -----------------------------
VECTORSTORE_PATH = BASE_DIR / "vectorstore"
DATA_PATH = BASE_DIR / "data"
METADATA_PATH = DATA_PATH / "metadata"

# -----------------------------
# Embedding Model
# (must match the model used to build the vectorstore - do not change
#  unless you rebuild the vectorstore)
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------
# NVIDIA LLM
# -----------------------------
# NVIDIA retires models often. Put the model that worked in test_nvidia.py
# on the DEFAULT line below (replace the name inside the quotes).
# To switch models later without touching code, add a line like
#   LLM_MODEL=nvidia/nemotron-3-super-120b-a12b
# to your .env file - it overrides the default.
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
LLM_MAX_TOKENS = 900

# -----------------------------
# Retrieval
# -----------------------------
SEARCH_TYPE = "mmr"        # "mmr" = diverse results, "similarity" = plain top-k
TOP_K = 5                  # chunks sent to the LLM (was 3)
FETCH_K = 20               # candidates considered before MMR picks TOP_K
MMR_LAMBDA = 0.6           # 1.0 = pure relevance, 0.0 = pure diversity

# ~1200-char chunks, so 6500 fits 5 chunks. (The old 3000 limit silently
# dropped the 3rd chunk.)
MAX_CONTEXT_CHARS = 6500

# Max related landmark cases (from cases.csv) added to the prompt
MAX_CASES = 3
