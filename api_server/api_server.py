from fastapi import FastAPI, Request, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
from gemma_unsloth import do_gemma_3n_inference, MODELS
import time
import logging
import json
import re
from fastapi.responses import StreamingResponse
import io
import numpy as np
import soundfile as sf

try:
    from kokoro import KPipeline
    print("Loading Kokoro TTS model...")
    # pipeline_tts = KPipeline(lang_code='a', device='cpu')
    pipeline_tts = KPipeline(lang_code='a', device='cuda')
    print("Kokoro TTS model loaded.")
    TTS_ENABLED = True
except ImportError:
    print("Kokoro TTS not found, TTS endpoint will be disabled.")
    TTS_ENABLED = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Create two routers, one for /v1 and one for the root
router_v1 = APIRouter(prefix="/v1")
router_root = APIRouter()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]], None] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = 'af_heart'


@router_v1.post("/tts")
async def text_to_speech(request: TTSRequest):
    if not TTS_ENABLED:
        raise HTTPException(status_code=501, detail="TTS is not enabled on the server.")
    
    logger.info(f"TTS request for text: {request.text}")
    
    try:
        # Split text into sentences for more natural TTS
        sentences = re.split(r'(?<=[.?!])\s+', request.text.strip())
        
        all_audio_chunks = []
        
        for sentence in sentences:
            if not sentence:
                continue
            
            generator = pipeline_tts(sentence, voice=request.voice)
            for i, (gs, ps, audio) in enumerate(generator):
                all_audio_chunks.append(audio)

        if not all_audio_chunks:
            raise HTTPException(status_code=500, detail="TTS failed to generate audio.")

        # Concatenate all audio chunks into a single numpy array
        full_audio = np.concatenate(all_audio_chunks)
        
        # Save the audio to an in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, full_audio, 24000, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        logger.error(f"Error in TTS generation: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during TTS generation: {e}")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    return response

@app.get("/")
async def root():
    return {"message": "Gemma API is running"}

def get_models_data():
    """Shared data for model listing endpoints"""
    models_list = []
    for model_id in MODELS.keys():
        models_list.append({
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
            "ready": True,
            "permissions": []
        })
    return {"data": models_list}

def parse_tool_calls(response_text: str) -> tuple[str, List[Dict[str, Any]]]:
    """
    Parse tool calls from model response
    Returns: (cleaned_content, tool_calls_list)
    """
    tool_calls = []
    cleaned_content = response_text
    
    # Look for JSON-like tool call patterns
    json_pattern = r'```json\s*([\s\S]+?)\s*```'
    
    matches = re.findall(json_pattern, response_text)
    
    if not matches:
        return cleaned_content, []

    for i, match in enumerate(matches):
        try:
            tool_info = json.loads(match)
            
            # The model could return a single tool call or a list of them
            if not isinstance(tool_info, list):
                tool_info = [tool_info]

            for tool_item in tool_info:
                if not (isinstance(tool_item, dict) and "name" in tool_item and "arguments" in tool_item):
                    continue
                
                name = tool_item.get("name")
                arguments = tool_item.get("arguments")

                if not name or arguments is None:
                    continue

                if isinstance(arguments, str):
                    try:
                        parsed_args = json.loads(arguments)
                    except json.JSONDecodeError:
                        parsed_args = {"input": arguments}
                else:
                    parsed_args = arguments

                # The model can be inconsistent. It might wrap arguments in 'properties'
                # or 'input', or it might return a stringified JSON. This logic
                # attempts to normalize the arguments.
                if isinstance(parsed_args, str):
                    try:
                        # Attempt to parse stringified JSON
                        parsed_args = json.loads(parsed_args)
                    except json.JSONDecodeError:
                        # If it's not JSON, treat it as a single string argument.
                        pass

                if isinstance(parsed_args, dict):
                    # Unwrap from 'properties' if it's the sole key
                    if 'properties' in parsed_args and len(parsed_args) == 1 and isinstance(parsed_args['properties'], dict):
                        parsed_args = parsed_args['properties']
                    # Unwrap from 'input' if it's the sole key
                    elif 'input' in parsed_args and len(parsed_args) == 1:
                         # The 'input' could be a string or a nested JSON string
                        if isinstance(parsed_args['input'], str):
                            try:
                                nested_args = json.loads(parsed_args['input'])
                                if isinstance(nested_args, dict):
                                    parsed_args = nested_args
                            except json.JSONDecodeError:
                                # Not a JSON string, so we leave it as is.
                                pass
                
                tool_call = {
                    "id": f"call_{int(time.time())}_{i}_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(parsed_args)
                    }
                }
                tool_calls.append(tool_call)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON from tool call block: {match}")
            continue
    
    if tool_calls:
        # Clean the content by removing the JSON blocks
        cleaned_content = re.sub(json_pattern, '', response_text, flags=re.DOTALL).strip()
    
    return cleaned_content, tool_calls

@router_v1.get("/models")
async def list_models_v1():
    return get_models_data()

@router_v1.get("/model/info")
async def get_model_info():
    return get_models_data()

@router_root.get("/models")
async def list_models_root():
    return get_models_data()

async def process_chat_completion(request: ChatCompletionRequest):
    """Shared logic for chat completion endpoints"""
    try:
        # Convert messages to the format expected by do_gemma_3n_inference
        messages = []
        for msg in request.messages:
            message_dict = {"role": msg.role}
            # Handle different content types, including base64 images
            if msg.content is None:
                message_dict["content"] = []
            elif isinstance(msg.content, str):
                message_dict["content"] = [{"type": "text", "text": msg.content}]
            elif isinstance(msg.content, list):
                processed_content = []
                for item in msg.content:
                    if item.get("type") == "image" and "image" in item:
                        # Convert to the expected format
                        processed_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{item['image']}"}
                        })
                    elif item.get("type") == "audio_url":
                        processed_content.append(item)
                    else:
                        processed_content.append(item)
                message_dict["content"] = processed_content
            else:
                message_dict["content"] = msg.content

            if msg.tool_calls:
                message_dict["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id
                
            messages.append(message_dict)
        
        # Add tool information to the prompt if tools are provided
        if request.tools:
            tool_descriptions = []
            for tool in request.tools:
                tool_desc = f"Tool: {tool.function.name}\nDescription: {tool.function.description}\nParameters: {json.dumps(tool.function.parameters)}"
                tool_descriptions.append(tool_desc)

            tool_prompt = f"Available tools:\n\n{chr(10).join(tool_descriptions)}\n\nUse tools by responding with JSON format: ```json\n{{\"name\": \"tool_name\", \"arguments\": \"arguments_string\"}}\n```"

            # Check if a system message already exists
            system_message_found = False
            for msg in messages:
                if msg["role"] == "system":
                    # Append tool info to existing system message
                    if isinstance(msg["content"], list):
                        # Append a new text part for the tool prompt
                        msg["content"].append({"type": "text", "text": f"\n\n{tool_prompt}"})
                    else:
                        # Handle the case where content is a simple string
                        msg["content"] += f"\n\n{tool_prompt}"
                    system_message_found = True
                    break
            
            if not system_message_found:
                # Add a new system message if none exists
                tool_system_msg = {
                    "role": "system",
                    "content": [{"type": "text", "text": tool_prompt}]
                }
                messages.insert(0, tool_system_msg)
        
        # Process messages to ensure they are compatible with the chat template.
        # Gemma's template can be strict about alternating user/assistant roles.
        
        # Convert 'tool' role to 'user' as the template might not support it.
        for msg in messages:
            if msg['role'] == 'tool':
                msg['role'] = 'user'

        # Merge consecutive messages of the same role
        if not messages:
            processed_messages = []
        else:
            processed_messages = [messages[0]]
            for i in range(1, len(messages)):
                current_msg = messages[i]
                last_msg = processed_messages[-1]
                
                if current_msg['role'] == last_msg['role']:
                    # Merge content
                    content1 = last_msg.get('content', [])
                    content2 = current_msg.get('content', [])
                    
                    if (isinstance(content1, list) and content1 and 
                        isinstance(content2, list) and content2 and
                        'text' in content1[0] and 'text' in content2[0]):
                        content1[0]['text'] += '\n' + content2[0]['text']
                    elif isinstance(content1, list) and isinstance(content2, list):
                        content1.extend(content2)
                    
                    # Merge tool_calls for assistant
                    if current_msg['role'] == 'assistant' and 'tool_calls' in current_msg:
                        if 'tool_calls' not in last_msg:
                            last_msg['tool_calls'] = []
                        last_msg['tool_calls'].extend(current_msg['tool_calls'])
                else:
                    processed_messages.append(current_msg)
            
        messages = processed_messages
        
        try:
            response_text = do_gemma_3n_inference(
                messages,
                model_name=request.model,
                max_new_tokens=request.max_tokens
            )
        except Exception as e:
            logger.error(f"Error in do_gemma_3n_inference: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
        if response_text is None:
            raise HTTPException(status_code=500, detail="Inference failed")
        
        # Parse potential tool calls from response
        cleaned_content, tool_calls = parse_tool_calls(response_text)
        
        # Build response message
        response_message = {
            "role": "assistant"
        }
        
        if tool_calls:
            response_message["tool_calls"] = tool_calls
            # If there are tool calls, content should be null or empty
            if cleaned_content.strip():
                response_message["content"] = cleaned_content
            else:
                response_message["content"] = None
        else:
            response_message["content"] = response_text
            
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "system_fingerprint": "fp_" + str(int(time.time())),
            "choices": [{
                "index": 0,
                "message": response_message,
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }],
            "usage": {
                "prompt_tokens": len(str(messages).split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(str(messages).split()) + len(response_text.split())
            }
        }
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise e

@router_v1.post("/chat/completions")
async def chat_completions_v1(request: ChatCompletionRequest):
    return await process_chat_completion(request)

@router_root.post("/chat/completions")
async def chat_completions_root(request: ChatCompletionRequest):
    return await process_chat_completion(request)

app.include_router(router_v1)
app.include_router(router_root)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
