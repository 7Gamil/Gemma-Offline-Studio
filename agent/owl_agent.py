# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Import from the correct module path

import os
import sys


from dotenv import load_dotenv
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.messages.base import BaseMessage

from camel.toolkits import (
    CodeExecutionToolkit,
    FileWriteToolkit,
)

from camel.types import ModelPlatformType, ModelType

from owl.utils import OwlRolePlaying
from typing import Dict, List, Optional, Tuple
from camel.logger import set_log_level, set_log_file, get_logger

import pathlib

logger = get_logger(__name__)

base_dir = pathlib.Path(__file__).parent.parent.parent
env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))

set_log_level(level="debug")
#set_log_level(level="NOTSET")

class ExcelRolePalying(OwlRolePlaying):
    def _construct_gaia_sys_msgs(self):
        user_system_prompt = f"""You are a user instructing an assistant to complete a task.
- Give one instruction at a time in the format: `Instruction: [YOUR INSTRUCTION]`.
- Do not attempt to solve the task in a single step. Break it down.
- Remind me to verify my work and run any code I write.
- Our overall task is: <task>{self.task_prompt}</task>.
- When the task is complete, reply only with <TASK_DONE>.
"""
        
        assistant_system_prompt = f"""You are an assistant helping a user complete a task.
- You must leverage your available tools, try your best to solve the problem, and explain your solutions.
- I have access to tools like code execution and file writing.
- [TOOL_CODE] must be a valid tool call from the available toolkits. To execute python code, you must provide the code in a 
- When you need to use a tool, you must use the following format and only this format:
JSON block as follows:
```json
{{
    "name": "execute_code",
    "arguments": {{
        "code": "from datetime import date\\n\\ntoday = date.today()\\nprint(f\'Today\\\'s date is: {{today}}\')",
        "code_type": "python"
    }}
}}
```
To execute a bash command, you must provide the command in a JSON block as follows:
```json
{{
    "name": "execute_code",
    "arguments": {{
        "code": "ls -l",
        "code_type": "bash"
    }}
}}
```
- Do not add any other text or formatting.
- Our overall task is: <task>{self.task_prompt}</task>.
- If a tool or code fails, debug and retry. Do not assume success.
- Verify your final answer.
"""
        
# To write to a file, you must provide the file path and content in a JSON block as follows:
# ```json
# {{
#     "name": "write_to_file",
#     "arguments": {{
#         "path": "test.txt",
#         "content": "This is a test."
#     }}
# }}
# ```

        user_sys_msg = BaseMessage.make_user_message(
            role_name=self.user_role_name, content=user_system_prompt
        )

        assistant_sys_msg = BaseMessage.make_assistant_message(
            role_name=self.assistant_role_name, content=assistant_system_prompt
        )

        return user_sys_msg, assistant_sys_msg


def run_society(
    society: ExcelRolePalying,
    round_limit: int = 15,
) -> Tuple[str, List[dict], dict]:
    overall_completion_token_count = 0
    overall_prompt_token_count = 0

    chat_history = []
    init_prompt = """
    Now please give me instructions to solve over overall task step by step. If the task requires some specific knowledge, please instruct me to use tools to complete the task.
        """
    input_msg = society.init_chat(init_prompt)
    for _round in range(round_limit):
        assistant_response, user_response = society.step(input_msg)
        # Check if usage info is available before accessing it
        if assistant_response.info.get("usage") and user_response.info.get("usage"):
            overall_completion_token_count += assistant_response.info["usage"].get(
                "completion_tokens", 0
            ) + user_response.info["usage"].get("completion_tokens", 0)
            overall_prompt_token_count += assistant_response.info["usage"].get(
                "prompt_tokens", 0
            ) + user_response.info["usage"].get("prompt_tokens", 0)

        # convert tool call to dict
        tool_call_records: List[dict] = []
        if assistant_response.info.get("tool_calls"):
            for tool_call in assistant_response.info["tool_calls"]:
                tool_call_records.append(
                    tool_call if isinstance(tool_call, dict) else tool_call.as_dict()
                )

        _data = {
            "user": user_response.msg.content
            if hasattr(user_response, "msg") and user_response.msg
            else "",
            "assistant": assistant_response.msg.content
            if hasattr(assistant_response, "msg") and assistant_response.msg
            else "",
            "tool_calls": tool_call_records,
        }

        chat_history.append(_data)
        logger.info(
            f"Round #{_round} user_response:\n {user_response.msgs[0].content if user_response.msgs and len(user_response.msgs) > 0 else ''}"
        )
        logger.info(
            f"Round #{_round} assistant_response:\n {assistant_response.msgs[0].content if assistant_response.msgs and len(assistant_response.msgs) > 0 else ''}"
        )
        #print(f"\nModel response: {assistant_response.msg.content}\n")
        print("===========USER==================")
        print(f"Raw user response: {user_response.info}")

        print("===========Assistant==================")
        print(f"Raw response: {assistant_response.info}")

        if (
            assistant_response.terminated
            or user_response.terminated
            or "TASK_DONE" in user_response.msg.content
        ):
            break

        input_msg = assistant_response.msg

    answer = chat_history[-1]["assistant"]
    token_info = {
        "completion_token_count": overall_completion_token_count,
        "prompt_token_count": overall_prompt_token_count,
    }

    return answer, chat_history, token_info

def construct_society(question: str) -> ExcelRolePalying:
    r"""Construct a society of agents based on the given question.

    Args:
        question (str): The task or question to be addressed by the society.

    Returns:
        OwlRolePlaying: A configured society of agents ready to address the question.
    """

    # Create models for different components using Azure OpenAI
    base_model_config = {
        "model_platform": ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        "model_type": "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
        #"model_type": "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
        "url": "http://0.0.0.0:8000/v1",
        "api_key": "not-needed",
        "model_config_dict": ChatGPTConfig(temperature=0.01, max_tokens=4096).as_dict()
    }

    model = ModelFactory.create(**base_model_config)

    # Configure toolkits
    tools = [
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *FileWriteToolkit("./").get_tools(),
    ]

    # Configure agent roles and parameters
    user_agent_kwargs = {"model": model, "tools": tools}
    assistant_agent_kwargs = {"model": model, "tools": tools}

    # Configure task parameters
    task_kwargs = {
        "task_prompt": question,
        "with_task_specify": False,
    }

    # Create and return the society
    society = ExcelRolePalying(
        **task_kwargs,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
        output_language="English"
    )

    return society


def main():
    # set_log_file('log.txt')

    # Override default task if command line argument is provided
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = input("Please enter the task: ")

    # Construct and run the society
    society = construct_society(task)

    answer, chat_history, token_count = run_society(society)

    # Output the result
    print(f"\033[94mAnswer: {answer}\033[0m")


if __name__ == "__main__":
    main()
