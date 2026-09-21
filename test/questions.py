"""Ask the example questions through the RAG API and save the results."""

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import requests
import yaml
from pydantic import BaseModel, ConfigDict, Field


ROOT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = ROOT_DIR / "_example_questions"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "_example_answers"


class Question(BaseModel):
	id: str
	question: str
	client_reference: int


class QuestionEntry(BaseModel):
	id: str
	question: str


class QuestionFile(BaseModel):
	client_reference: int
	questions: list[QuestionEntry]


class QueryRequest(BaseModel):
	query: str
	scope: Literal["client"] = "client"
	client_reference: int


class TokenUsage(BaseModel):
	prompt_tokens: int = 0
	completion_tokens: int = 0

	@property
	def total_tokens(self) -> int:
		return self.prompt_tokens + self.completion_tokens


class Match(BaseModel):
	model_config = ConfigDict(extra="ignore")

	file_name: str = ""
	context: str = ""
	score: float | None = None


class ContextResponse(BaseModel):
	response: str = ""
	token_usage: TokenUsage = Field(default_factory=TokenUsage)


class MatchData(BaseModel):
	match: Match
	context_response: ContextResponse


class QueryResponse(BaseModel):
	response: str = ""
	matches: list[MatchData] = Field(default_factory=list)


class Citation(BaseModel):
	source: str
	context: str
	score: float | None = None
	extracted_answer: str
	token_usage: TokenUsage


class TokenCosts(BaseModel):
	prompt_tokens: int
	completion_tokens: int
	total_tokens: int


class Answer(BaseModel):
	id: str
	question: str
	answer: str
	citations: list[Citation]
	token_costs: TokenCosts


class AnswerFile(BaseModel):
	answers: list[Answer]


def load_questions(path: Path) -> Iterator[tuple[str, list[Question]]]:
	"""Yield each question file name together with its questions."""
	question_files = [path] if path.is_file() else sorted(path.glob("*.yaml"))

	for question_file in question_files:
		with question_file.open(encoding="utf-8") as file:
			question_set = QuestionFile.model_validate(yaml.safe_load(file) or {})
		questions = [
			Question(
				id=question.id,
				question=question.question,
				client_reference=question_set.client_reference,
			)
			for question in question_set.questions
		]

		yield question_file.name, questions


def make_query(api_url: str, question: Question, timeout: float) -> QueryResponse:
	"""Send one client-scoped question to the RAG API."""
	query_request = QueryRequest(
		query=question.question,
		client_reference=question.client_reference,
	)
	response = requests.post(
		f"{api_url.rstrip('/')}/query",
		json=query_request.model_dump(),
		timeout=timeout,
	)
	response.raise_for_status()
	return QueryResponse.model_validate(response.json())


def format_result(question: Question, result: QueryResponse) -> Answer:
	"""Keep the answer file focused on answers, citations, and token usage."""
	citations: list[Citation] = []
	prompt_tokens = 0
	completion_tokens = 0

	for match_data in result.matches:
		match = match_data.match
		context_response = match_data.context_response
		token_usage = context_response.token_usage
		prompt_tokens += token_usage.prompt_tokens
		completion_tokens += token_usage.completion_tokens
		citations.append(
			Citation(
				source=match.file_name,
				context=match.context,
				score=match.score,
				extracted_answer=context_response.response,
				token_usage=token_usage,
			)
		)

	return Answer(
		id=question.id,
		question=question.question,
		answer=result.response,
		citations=citations,
		token_costs=TokenCosts(
			prompt_tokens=prompt_tokens,
			completion_tokens=completion_tokens,
			total_tokens=prompt_tokens + completion_tokens,
		),
	)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--api-url", default="http://127.0.0.1:8000")
	parser.add_argument("--questions", type=Path, default=QUESTIONS_DIR)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--timeout", type=float, default=300)
	args = parser.parse_args()

	args.output_dir.mkdir(parents=True, exist_ok=True)
	total_answers = 0
	total_files = 0
	for file_name, questions in load_questions(args.questions):
		answers = []
		for index, question in enumerate(questions, 1):
			print(f"[{file_name} {index}/{len(questions)}] {question.id}: {question.question}")
			result = make_query(args.api_url, question, args.timeout)
			answers.append(format_result(question, result))

		output_file = args.output_dir / file_name
		with output_file.open("w", encoding="utf-8") as file:
			yaml.safe_dump(
			AnswerFile(answers=answers).model_dump(),
			file,
			sort_keys=False,
			allow_unicode=True,
		)
		total_answers += len(answers)
		total_files += 1
		print(f"Saved {len(answers)} answers to {output_file}")

	print(f"Saved {total_answers} answers from {total_files} files")


if __name__ == "__main__":
	main()