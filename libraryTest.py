import numpy
import torch
from sentence_transformers import SentenceTransformer

print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
model = SentenceTransformer("all-MiniLM-L6-v2")
print("MODEL LOAD OK")
