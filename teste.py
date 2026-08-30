from llm_sdk import Small_LLM_Model
import json

llm = Small_LLM_Model()

with open(llm.get_path_to_vocab_file(), "r", encoding="UTF-8") as f:
    vocab = json.load(f)

print(vocab)
