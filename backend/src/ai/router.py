from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict
from .chatgpt_client import ChatGPTClient
from .config import get_ai_settings

router = APIRouter(prefix="/ai", tags=["ai"])

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class SinglePromptRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class ChatResponse(BaseModel):
    response: str

@router.post("/chat", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    settings: AISettings = Depends(get_ai_settings)
) -> ChatResponse:
    """
    Get a completion from ChatGPT using a list of messages
    """
    client = ChatGPTClient()
    try:
        response = await client.get_completion(
            request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/prompt", response_model=ChatResponse)
async def single_prompt(
    request: SinglePromptRequest,
    settings: AISettings = Depends(get_ai_settings)
) -> ChatResponse:
    """
    Get a completion from ChatGPT using a single prompt
    """
    client = ChatGPTClient()
    try:
        response = await client.get_single_completion(
            request.prompt,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
