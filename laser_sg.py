import os
import time
import json
import argparse
from langchain_openai import ChatOpenAI
from laser.scene_generator.script_writer import ScriptWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script Generation')
    parser.add_argument("-s", "--scene", 
                        type=str, 
                        help="name of the scene, we seek UserPrompt.txt from ./laser_scenes/name_of_the_scene/", required=True)
    args = parser.parse_args()

    scene = args.scene

    root_directory = os.path.join('laser_scenes', scene)
    user_prompt_path = os.path.join(root_directory, 'UserRequirement.txt')

    llm = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
    script_writer = ScriptWriter(ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"]), user_prompt_path)
    sub_scripts = script_writer.get() # json dict
    script = script_writer.script

    with open(os.path.join(root_directory, 'output.txt'), 'w') as file:
        file.write(script)
        os.chmod(os.path.join(root_directory, 'output.txt'), 0o444)

    with open(os.path.join(root_directory, 'script.json'), 'w') as file:
        json.dump(sub_scripts, file, indent=4)

    print(f"script written to: {os.path.join(root_directory, 'script.json')}")
