from llm_sdk import Small_LLM_Model


llm = Small_LLM_Model()

# pergunta = "What is the sum of 2 and 3?"
# pergunta = "Greet shrek"
pergunta = "Reverse the string 'hello'"

prompt = f"""
Available functions:
fn_add_numbers(a: number, b: number)
fn_greet(name: string)
fn_reverse_string(s: string)

User request:
{pergunta}

Function call:
The function call would be
"""

ids = llm.encode(prompt)[0].tolist()

generated_ids = []

while ")" not in llm.decode(generated_ids) and len(generated_ids) < 50:
    logits = llm.get_logits_from_input_ids(ids)

    next_token_id = logits.index(max(logits))

    ids.append(next_token_id)
    generated_ids.append(next_token_id)

print(llm.decode(generated_ids).strip())
