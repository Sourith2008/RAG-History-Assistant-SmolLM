from app.model import model,tokenizer,embed_model
import faiss
import pickle
import numpy as np
import torch

with open("data/docs.pkl","rb") as f:
    docs=pickle.load(f)
index=faiss.read_index("data/index.faiss")
@torch.inference_mode()
def infer(user_input):
  model.eval()
  top_k=5
  query_embedding=embed_model.encode([user_input])
  query_embedding=(query_embedding).astype('float32')
  faiss.normalize_L2(query_embedding)
  distances,indices=index.search(query_embedding,top_k)
  contexts=[docs[i] for i in indices[0]]
  top_contexts=contexts[:2]
  context="\n\n".join(top_contexts)
  messages = [
    {
        "role": "system",
        "content":"""
You are a Class IX History assistant.

Answer the question using ONLY the provided context.

Rules:
- Answer in exactly one sentence whenever possible.
- Be direct and concise.
- Do not explain unnecessarily.
- Do not repeat the question.
- Do not create lists unless the user explicitly asks for a list.
- Do not use information outside the provided context.
- If the context does not contain the answer, say:
  "The answer is not available in the provided text."
"""

     },
    {
        "role":"user",
        "content":f"""
        context:
        {context}

        Question:
        {user_input}
        """
    }
]
  inputs = tokenizer.apply_chat_template(
  messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
  )
  outputs=model.generate(
      **inputs,
      max_new_tokens=150
  )
  length=inputs["input_ids"].shape[-1]
  response=tokenizer.decode(outputs[0][length:],skip_special_tokens=True)
  return response.strip()
