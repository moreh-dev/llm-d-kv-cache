# Copyright 2025 The llm-d Authors.
#
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

#!/usr/bin/env python3
"""
Standalone wrapper for tokenizer from vllm.
"""

import json
import logging
import os
import sys

from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.plugins.io_processors import get_io_processor
from vllm.entrypoints.openai.api_server import (
    build_app,
    init_app_state,
)
import asyncio

from vllm.v1.engine.input_processor import InputProcessor

# Basic logging setup
logger = logging.getLogger(__name__)


def _create_parser():
    """Create a new argument parser instance. Thread-safe as each call creates a new parser."""
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    return make_arg_parser(parser)


class EngineClientMock:
    """
    Lightweight mock of vLLM's AsyncLLMEngineClient for tokenization-only operations.

    This class provides the minimal interface required by vLLM's OpenAI-compatible
    serving layer (openai_serving_chat, openai_serving_completion) to perform
    tokenization and chat template rendering without running a full inference engine.

    Why a mock instead of the real engine client:
    - The real AsyncLLMEngineClient requires GPU resources and model loading
    - We only need tokenization/chat template functionality, not inference
    - This allows CPU-only operation for the preprocessing pipeline

    Replaces: vllm.entrypoints.openai.engine.async_llm_engine.AsyncLLMEngineClient

    Limitations:
    - Only supports 'generate' task (no embeddings, classifications, etc.)
    - Does not perform actual inference - only tokenization and prompt rendering
    - The 'errored' flag is always False as there's no engine to fail

    Used by: init_app() to initialize vLLM's serving components
    """

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config
        self.input_processor = InputProcessor(self.vllm_config)
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.vllm_config.model_config.io_processor_plugin,
        )
        self.model_config = vllm_config.model_config
        self.errored = False
        self.renderer = self.input_processor.renderer

    async def get_supported_tasks(self):
        return ("generate",)

    async def get_tokenizer(self):
        return self.input_processor.get_tokenizer()


_app = None
_loop = None


def _run_async(coro):
    """Run async coroutine using a persistent event loop."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop.run_until_complete(coro)


def clear_caches():
    """Clear the tokenizer cache for testing purposes."""
    global _app
    _app = None
    return "Tokenizer caches cleared"


def init_app(request_json):
    """
    Initialize the vLLM app with the specified tokenizer configuration.
    If the app is already initialized, this is a no-op.

    Args:
        request_json (str): JSON string containing the request parameters:
            - model (str): The model ID or path (HF model ID, local directory path, or path to tokenizer file).
            - is_local (bool, optional): Whether the model is local.
            - tokenizer (str, optional): Tokenizer path (defaults to model path).
            - tokenizer_mode (str, optional): Tokenizer mode (default: auto).
                - "auto": use mistral_common for Mistral models if available, otherwise "hf".
                - "hf": use the fast tokenizer if available.
                - "slow": always use the slow tokenizer.
                - "mistral": always use the tokenizer from mistral_common.
                - "deepseek_v32": always use the tokenizer from deepseek_v32.
                - Other custom values can be supported via plugins.
            - revision (str, optional): Model revision.
            - tokenizer_revision (str, optional): Tokenizer revision.
            - token (str, optional): Hugging Face token for private models.
            - download_dir (str, optional): Directory to download the model.

    Note:
        Setting is_local=True does NOT prevent downloading if the model path is not a file or directory.
        For example, if is_local=True but model is a HuggingFace model ID, it will still be downloaded.
        Conversely, if is_local=False but model is a file or directory path, the model will NOT be downloaded and will be loaded locally.
    """
    # Parse the JSON request
    request = json.loads(request_json)

    try:
        global _app
        if _app is not None:
            return
        model_name = request.pop("model")
        revision = request.get("revision", None)
        is_local = request.pop("is_local", False)
        token = request.pop("token", "")
        download_dir = request.pop("download_dir", None)
        tokenizer = request.pop("tokenizer", None)
        tokenizer_mode = request.pop("tokenizer_mode", "auto")
        tokenizer_revision = request.pop("tokenizer_revision", None)

        if is_local and os.path.isfile(model_name):
            # If it's a file path (tokenizer.json), get the directory
            model_name = os.path.dirname(model_name)

        # Create a new parser instance for thread-safety, and pass empty list
        # to avoid parsing sys.argv (which may contain unrelated arguments)
        args = _create_parser().parse_args([])
        args.model = model_name
        args.hf_token = token
        args.download_dir = download_dir
        args.tokenizer = tokenizer
        args.tokenizer_mode = tokenizer_mode
        args.revision = revision
        args.tokenizer_revision = tokenizer_revision
        args.trust_request_chat_template = True
        engine_args = AsyncEngineArgs.from_cli_args(args)
        vllm_config = engine_args.create_engine_config()

        engine_client = EngineClientMock(vllm_config)
        _app = build_app(args)
        # Note: init_app_state triggers a warmup that may log "Chat template warmup failed"
        # if the model doesn't have a default chat template (e.g., facebook/opt-125m).
        # This error can be safely ignored - actual render_chat calls will work correctly
        # because they pass chat_template in the request.
        _run_async(init_app_state(engine_client, _app.state, args))
    except Exception as e:
        raise RuntimeError(
            f"Error initializing tokenizer ({type(e).__name__}): {e}"
        ) from e


def render_chat(request_json):
    """
    Render a chat template using the vllm library.
    This function is aligned with the Go cgo_functions.go structs.

    Args:
        request_json (str): JSON string containing the request parameters:
            - conversation (list): List of message dicts, each with 'role' and 'content' keys
            - chat_template (str, optional): The template to use
            - tools (list, optional): Tool schemas
            - documents (list, optional): Document schemas
            - return_assistant_tokens_mask (bool, optional): Whether to return assistant tokens mask
            - continue_final_message (bool, optional): Whether to continue final message
            - add_generation_prompt (bool, optional): Whether to add generation prompt
            - chat_template_kwargs (dict, optional): Additional rendering variables

    Returns:
        JSON string containing:
            - input_ids (list of int): The list of token IDs.
            - offset_mapping (list): Always empty. Offset mappings are not supported
              by vLLM's render_chat_request API.
    """

    try:
        global _app
        # Parse the JSON request
        request = json.loads(request_json)
        if _app is None:
            raise RuntimeError("App not found in cache")

        # Get template_vars and spread them as individual arguments
        template_vars = request.pop("chat_template_kwargs", {})
        request.update(template_vars)

        # Remove model since it's already set in the app state
        request.pop("model", None)

        # Convert conversation to messages for ChatCompletionRequest
        if "conversation" in request:
            request["messages"] = request.pop("conversation")

        result = _run_async(
            _app.state.openai_serving_chat.render_chat_request(
                ChatCompletionRequest(**request)
            )
        )
        if isinstance(result, ErrorResponse):
            raise RuntimeError(f"Chat template error: {result.error.message}")
        # result is tuple of (conversation, engine_prompts)
        # engine_prompts[0] contains prompt_token_ids
        _, engine_prompts = result
        if not engine_prompts or len(engine_prompts) == 0:
            raise RuntimeError("render_chat_request returned empty engine_prompts")
        # Convert to match docstring format
        return json.dumps(
            {
                "input_ids": engine_prompts[0].get("prompt_token_ids", []),
                "offset_mapping": [],
            }
        )

    except Exception as e:
        raise RuntimeError(
            f"Error applying chat template ({type(e).__name__}): {e}"
        ) from e


def render(request_json: str) -> str:
    """
    Render text using the specified tokenizer.

    Args:
        request_json (str): JSON string containing:
            - text (str): The text to render
            - add_special_tokens (bool, optional): Whether to add special tokens

    Returns:
        JSON string containing:
            - input_ids (list of int): The list of token IDs.
            - offset_mapping (list): Always empty. Offset mappings are not supported
              by vLLM's render_completion_request API.
    """
    try:
        global _app
        # Parse the JSON request
        request = json.loads(request_json)
        text = request["text"]
        add_special_tokens = request.get("add_special_tokens", False)
        if _app is None:
            raise RuntimeError("App not found in cache")

        result = _run_async(
            _app.state.openai_serving_completion.render_completion_request(
                CompletionRequest(
                    prompt=text,
                    add_special_tokens=add_special_tokens,
                )
            )
        )
        if isinstance(result, ErrorResponse):
            raise RuntimeError(f"Completion render error: {result.error.message}")
        # result is list of dicts with prompt_token_ids
        if not result or len(result) == 0:
            raise RuntimeError("render_completion_request returned empty result")
        # Convert to match docstring format
        return json.dumps(
            {
                "input_ids": result[0].get("prompt_token_ids", []),
                "offset_mapping": [],
            }
        )

    except Exception as e:
        raise RuntimeError(f"Error rendering text ({type(e).__name__}): {e}") from e


# python pkg/preprocessing/chat_completions/tokenizer_wrapper.py True '{"model": "/mnt/models/hub/models--ibm-granite--granite-3.3-8b-instruct/snapshots/51dd4bc2ade4059a6bd87649d68aa11e4fb2529b", "conversation": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "who are you?"}]}'
def main():
    """Example usage and testing function."""
    is_local = False
    if len(sys.argv) > 1:
        is_local = sys.argv[1].lower() == "true"

    # Default body if none provided
    body = {
        "model": "facebook/opt-125m",
        "conversation": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "who are you?"},
        ],
        "chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}",
    }
    if len(sys.argv) > 2:
        body = json.loads(sys.argv[2])

    try:
        # Construct the request JSON string similar to how Go would
        init_app(
            json.dumps(
                {
                    "is_local": is_local,
                    "model": body.get("model"),
                }
            )
        )
        chat_request_str = json.dumps(body)
        render_chat_result = render_chat(chat_request_str)
        print(render_chat_result)
        last_content = body["conversation"][-1]["content"]
        if isinstance(last_content, str):
            render_request = {
                "text": last_content,
                "add_special_tokens": True,
            }
            render_request_str = json.dumps(render_request)
            render_result = render(render_request_str)
            print(render_result)
        else:
            print(
                "Skipping render(): multimodal content is not supported by CompletionRequest"
            )
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
