from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, Field, PrivateAttr
import json
import time
#TODO: Ainda preciso ver se vou implantar outras llm no meu modelo


class Function_Calling(BaseModel):
    model_name: str = Field(default="Qwen/Qwen3-0.6B")
    path_fn_definition: str = Field(
        default="./data/input/functions_definition.json"
    )
    _llm: Small_LLM_Model = PrivateAttr(default_factory=Small_LLM_Model)
    _token_cache: dict = PrivateAttr(default_factory=dict)
    _function_names: set[str] = PrivateAttr(default_factory=set)
    _function_prefixes: set[str] = PrivateAttr(default_factory=set)

    def model_post_init(self, __context) -> None:
        self._function_names = self.get_functions_name()
        self._function_prefixes = self.get_prefixes()

    def get_functions_name(self) -> set:
        #TODO: Colocar um try aqui caso não abra
        with open(self.path_fn_definition, "r", encoding="utf-8") as f:
            function_names = [x["name"] for x in json.load(f)]
        return set(function_names)

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

        Use each function's description to determine which function is
        appropriate.

        Return only the exact function name.
        Do not include explanations, comments, markdown, arguments, or any
        other text.

        Function name:
        "
        """
        return sys_prompt

    def get_prefixes(self) -> None:
        self._function_prefixes = {
            name[:i]
            for name in self._function_names
            for i in range(1, len(name) + 1)
        }

        return self._function_prefixes

    def get_output_name(self, user_prompt: str) -> str:
        def is_prefix(candidate: str) -> bool:
            if "\"" in candidate:
                candidate = candidate.split("\"", 1)[0]
                if candidate not in self._function_names:
                    return False
            if candidate not in self._function_prefixes:
                return False
            return True

        ids: list[int] = self._llm.encode(
            self.get_sys_prompt(user_prompt)
        )[0].tolist()

        name: str = ""
        while True:

            logits: list[float] = self._llm.get_logits_from_input_ids(ids)

            while True:
                name: str

                for token_id in range(len(logits)):
                    if token_id not in self._token_cache:
                        decoded_token = self._llm.decode([token_id])
                        self._token_cache[token_id] = decoded_token

                    candidate = name + self._token_cache[token_id]
                    if not is_prefix(candidate):
                        logits[token_id] = float("-inf")

                next_token_id: int = logits.index(max(logits))

                next_piece = self._token_cache[next_token_id]

                if "\"" in next_piece:
                    before_quote = next_piece.split("\"", 1)[0]
                    candidate = name + before_quote

                    if candidate in self._function_names:
                        name = candidate

                if name + next_piece in self._function_prefixes:
                    name += next_piece

                    if (
                        name in self._function_names
                        and sum(
                            function_name.startswith(name)
                            for function_name in self._function_names
                        ) == 1
                    ):
                        break

                    ids.append(next_token_id)
                    break

            if name in self._function_names:
                possible = sum(
                    function_name.startswith(name)
                    for function_name in self._function_names
                )

                if possible == 1:
                    break

        return name

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

start = time.time()
print(f'Start time: {start}')

with open("./data/input/function_calling_tests.json") as f:
    prompts: list[str] = [x["prompt"] for x in json.load(f)]

llm = Function_Calling()
for x in prompts:
    print(llm.output(x))

end = time.time()

elapsed = end - start

minutes = int(elapsed // 60)
seconds = elapsed % 60

print(f"Elapsed: {minutes}m {seconds:.2f}s")
