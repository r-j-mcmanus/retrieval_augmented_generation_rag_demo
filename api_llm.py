from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pydantic_dataclasses import LLMResponse

from llm_caller.base import LLMCallerInterface
from llm_caller.local_qwen import LocalQwenLLMCaller


LLM_CALLER: LLMCallerInterface | None  = None
GENERATION_LOCK = Lock()

# look at models in https://www.reddit.com/r/LocalLLaMA/comments/1wg28mu/is_there_a_better_small_model_than_qwen35_4b_for/


class GenerateRequest(BaseModel):
	query: str


@asynccontextmanager
async def lifespan(_: FastAPI):
	initialize_model()
	yield


app = FastAPI(title="Local LLM API", version="1.0.0", lifespan=lifespan)


def initialize_model() -> None:
	global LLM_CALLER
	LLM_CALLER = LocalQwenLLMCaller()


def generate_reply(prompt: str) -> LLMResponse:
	if LLM_CALLER is None:
		initialize_model()
	assert isinstance(LLM_CALLER, LLMCallerInterface)
	answer = LLM_CALLER.call(prompt)
	return answer

@app.post("/generate", response_model=LLMResponse)
def generate(request: GenerateRequest) -> LLMResponse:
	try:
		with GENERATION_LOCK:
			response = generate_reply(request.query)
	except Exception as error:
		raise HTTPException(status_code=503, detail=f"Model unavailable: {error}") from error
	return response
