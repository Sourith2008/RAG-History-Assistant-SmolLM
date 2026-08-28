from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch
import streamlit as st

@st.cache_resource
def load_model():
    model=AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct",)
    tokenizer=AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

    embed_model=SentenceTransformer("all-MiniLM-L6-v2")
    
    model.eval()
    return model,tokenizer,embed_model

model,tokenizer,embed_model=load_model()