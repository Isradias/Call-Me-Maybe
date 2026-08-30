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
        self.load_functions()

    def load_functions(self) -> None:
        #TODO: Colocar um try aqui
        with open(self.path_fn_definition, "r", encoding="utf-8") as f:
            self._functions_definition = json.load(f)

        self._function_names = {
            function["name"]
            for function in self._functions_definition
        }

        self._function_prefixes = {
            name[:i]
            for name in self._function_names
            for i in range(1, len(name) + 1)
        }

    def get_sys_prompt(self, prompt: str, name: str | None = None) -> str:
        if not name:
            sys_prompt = (f"""
            You are a function-calling system.

            Available functions:
            {self._functions_definition}

            User request:
            {prompt}

            Select exactly one function whose described behavior best matches
            the user's request.

            Prefer the simplest function that fully satisfies the request.
            Do not choose a more specific function unless the user explicitly
            requests the additional behavior described by that function.

            Use each function's description to determine the correct function.

            Return only the exact function name.
            Do not include explanations, comments, markdown, arguments, or any
            other text.

            Function name:
            "
            """).strip()
        else:
            function_definition = next(
                x for x in self._functions_definition
                if x["name"] == name
            )

            sys_prompt = f"""
            You are extracting arguments for a function call.

            Selected function:
            {function_definition}

            User request:
            {prompt}

            Extract exactly the parameters required by the selected function
            from the user request.

            Use the parameter names and types defined by the function.
            Do not invent additional parameters.
            Do not include explanations, comments, markdown, or any other text.

            Parameters:
            {{
            """.strip()
        return sys_prompt

    def get_prefixes(self) -> None:
        self._function_prefixes = {
            name[:i]
            for name in self._function_names
            for i in range(1, len(name) + 1)
        }

        return self._function_prefixes

    def get_output_name(self, user_prompt: str) -> str:
        def is_valid_candidate(candidate: str, name: str) -> bool:
            if candidate == name:
                return False
            if "\"" in candidate:
                candidate = candidate.split("\"", 1)[0]
                if candidate not in self._function_names:
                    return False
            if candidate not in self._function_prefixes:
                return False
            return True

        def is_unique_function(name: str):
            functions = [function_name.startswith(name)
                         for function_name in self._function_names]
            nb_functions = sum(functions)
            if nb_functions == 1 and name in self._function_names:
                return True
            return False

        ids: list[int] = self._llm.encode(
            self.get_sys_prompt(user_prompt)
        )[0].tolist()

        name: str = ""
        while True:

            logits: list[float] = self._llm.get_logits_from_input_ids(ids)

            for token_id in range(len(logits)):
                if token_id not in self._token_cache:
                    decoded_token = self._llm.decode([token_id])
                    self._token_cache[token_id] = decoded_token

                candidate = name + self._token_cache[token_id]

                if not is_valid_candidate(candidate, name):
                    logits[token_id] = float("-inf")

            next_token_id: int = logits.index(max(logits))

            argmax = self._token_cache[next_token_id]

            if "\"" in argmax:
                before_quote = argmax.split("\"", 1)[0]
                return name + before_quote

            name += argmax

            if is_unique_function(name):
                return name

            ids.append(next_token_id)

    def get_output_parameters(self, user_prompt: str, name: str):
        def get_nb_parameters(name):
            function_definition = next(
                            x for x in self._functions_definition
                            if x["name"] == name
                        )
            return len(function_definition["parameters"])

        def get_parameters(name) -> dict:
            function_definition = next(
                            x for x in self._functions_definition
                            if x["name"] == name
                        )
            return function_definition["parameters"]

        prompt = self.get_sys_prompt(user_prompt, name)
        nb_parameters = get_nb_parameters(name)
        
        for parameter in get_parameters(name):
            prompt += "\"" + parameter + "\": \""
            prompt += "exemplo" #TODO: É aqui que entrarão as escolhas da llm
            prompt += "\""
            if nb_parameters > 1:
                prompt += ", "
                nb_parameters -= 1
            print(prompt)
        return 

    def output(self, user_prompt: str) -> str:
        prompt = user_prompt
        name = self.get_output_name(prompt)
        parameters = self.get_output_parameters(prompt, name)

        data = {
            "prompt": prompt,
            "name": name,
            "parameters": parameters,
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
