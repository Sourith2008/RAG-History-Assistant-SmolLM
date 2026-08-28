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
- Give a direct answer.
- Use 2 to 4 sentences.
- Do not add information that is not present in the context.
- Do not guess.
- If the context does not contain the answer, say:
  "The answer is not available in the provided text."
- Do not create lists unless the question asks for a list.
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
      max_new_tokens=100
  )
  length=inputs["input_ids"].shape[-1]
  response=tokenizer.decode(outputs[0][length:],skip_special_tokens=True)
  return response.strip()
