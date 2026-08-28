from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, Field, PrivateAttr
import json
#TODO: Ainda preciso ver se vou implantar outras llm no meu modelo


class Function_Calling(BaseModel):
    model_name: str = Field(default="Qwen/Qwen3-0.6B")
    path_fn_definition: str = Field(
        default="./data/input/functions_definition.json"
    )
    _llm: Small_LLM_Model = PrivateAttr(default_factory=Small_LLM_Model)

    def get_functions_name(self) -> list[str]:
        #TODO: Colocar um try aqui caso não abra
        with open(self.path_fn_definition, "r", encoding="utf-8") as f:
            function_names = [x["name"] for x in json.load(f)]
        return function_names

    def get_functions_definition(self) -> str:
        with open(self.path_fn_definition, "r", encoding="utf-8") as f:
            functions_definition = f.read()
        return functions_definition

    def get_sys_prompt(self, user_prompt: str) -> str:
        sys_prompt = f"""
        You are a function-calling system.

        Available functions:
        {self.get_functions_definition()}

        User request:
        {user_prompt}

        Select the single function that best matches the user request.

        Use each function's description to determine which function is appropriate.

        Return only the exact function name.
        Do not include explanations, comments, markdown, arguments, or any other text.

        Function name:
        "
        """
        return sys_prompt

    def get_output_name(self, user_prompt: str) -> str:
        ids: list[int] = self._llm.encode(
            self.get_sys_prompt(user_prompt)
        )[0].tolist()

        functions_name = self.get_functions_name()

        name: str = ""

        while True:
            logits: list[float] = self._llm.get_logits_from_input_ids(ids)

            while True:
                next_token_id: int = logits.index(max(logits))
                next_piece: str = self._llm.decode([next_token_id])

                if "\"" in next_piece:
                    before_quote = next_piece.split("\"", 1)[0]
                    candidate = name + before_quote

                    if candidate in functions_name:
                        return candidate

                    logits[next_token_id] = float("-inf")
                    continue

                if any(
                    function_name.startswith(name + next_piece)
                    for function_name in functions_name
                ):
                    name += next_piece
                    ids.append(next_token_id)
                    break

                logits[next_token_id] = float("-inf")

    def output(self, user_prompt: str) -> str:
        data = {
            "prompt": user_prompt,
            "name": self.get_output_name(user_prompt),
            "parameters": "ainda não",
        }

        return json.dumps(
            data,
            ensure_ascii=False,
            indent=4,
        )


# template = f"""
#             {{
#                 "prompt": "{user_prompt}",
#                 "name": "{function_called}",
#                 "parameters": {{{parameters}}}
#             }}
#           """

with open("./data/input/function_calling_tests.json") as f:
    prompts: list[str] = [x["prompt"] for x in json.load(f)]

llm = Function_Calling()
for x in prompts:
    print(llm.output(x))
