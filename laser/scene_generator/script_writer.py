import os
import re
import json
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

PROMPTS_DIRECTORY = "laser/scene_generator/"


def extract_json(text):
    # Use regex to find the JSON part within triple backticks
    match = re.search(r'```json(.*?)```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()  # Extract and clean up the JSON part
        try:
            # Parse the JSON string into a Python dictionary
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return None
    else:
        print("No JSON found")
        return None


class ScriptWriter:
    def __init__(self, llm, user_prompt_path):
        self._llm = llm
        self.user_prompt_path = user_prompt_path
        
    def get(self):
        system_prompt = SystemMessagePromptTemplate(
            prompt = PromptTemplate.from_file(
                os.path.join(PROMPTS_DIRECTORY, "SystemPrompt.txt"), 
            )
        )
        user_prompt = HumanMessagePromptTemplate(
            prompt = PromptTemplate.from_file(
                self.user_prompt_path, 
            )
        )

        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            user_prompt,
        ])

        chain = prompt | self._llm

        self.output = chain.invoke({})
        self.script = self.output.content
        self.sub_scripts = self.post_process(self.output.content)
        self.usage_metadata = self.output.usage_metadata

        return self.sub_scripts

    def post_process(self, output):
        return extract_json(output)

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
    scene = "merge-alternately"
    user_prompt_path = os.path.join(PROMPTS_DIRECTORY, scene, "UserPrompt.txt")
    sw = ScriptWriter(llm, user_prompt_path)
    print(sw.get())