from pathlib import Path
import time
import requests

import streamlit as st
from pydantic_dataclasses import QueryRequest
from pydantic_dataclasses import DocumentScope

# run with streamlit run dashboard.py

DATA_DIR = Path(__file__).parent / "_data"

def get_data_files() -> list[Path]:
	"""Return files beneath _data in a stable display order."""
	if not DATA_DIR.exists():
		return []
	return sorted(
		(path for path in DATA_DIR.rglob("*") if path.is_file()),
		key=lambda path: path.relative_to(DATA_DIR).as_posix().lower(),
	)

def answer_query(query: str, scope_selection: str | int | None):
	client_reference = None
	if scope_selection == DocumentScope.INTERNAL:
		scope = DocumentScope.INTERNAL
	elif scope_selection == DocumentScope.ALL_CLIENTS:
		scope = DocumentScope.ALL_CLIENTS
	elif isinstance(scope_selection, int):
		scope = DocumentScope.CLIENT
		client_reference = scope_selection
	else:
		raise ValueError(f'Invalid scope {scope_selection}')

	started_at = time.perf_counter()
	response = requests.post(
		"http://127.0.0.1:8000/query",
		json=QueryRequest(
			query=query, 
			scope=scope,
			client_reference=client_reference
		).model_dump(),
	)
	response.raise_for_status()
	result = response.json()
	result["response_time_seconds"] = time.perf_counter() - started_at
	return result


def get_token_usage(result: dict) -> tuple[int, int]:
	"""Return prompt and completion tokens used by the complete query."""
	prompt_tokens = 0
	completion_tokens = 0

	responses = [result.get("response", {})]
	responses.extend(
		match_data.get("context_response", {})
		for match_data in result.get("matches", [])
	)
	for response in responses:
		usage = response.get("token_usage", {}) if isinstance(response, dict) else {}
		prompt_tokens += usage.get("prompt_tokens", 0)
		completion_tokens += usage.get("completion_tokens", 0)

	return prompt_tokens, completion_tokens


# -----------------------------------------
# -----------------------------------------
# -----------------------------------------


st.set_page_config(page_title="RAG Dashboard", page_icon="📚", layout="wide")
st.title("RAG Dashboard")

data_files = get_data_files()
selected_files: list[Path] = []

with st.sidebar:
	st.header("Settings")

	top_k = st.slider(
		"Max number of citations",
		min_value=1,
		max_value=20,
		value=3,
	)

	st.header("Data files")
	if not data_files:
		st.info("No files found in the _data folder.")
	else:
		for file_path in data_files:
			relative_path = file_path.relative_to(DATA_DIR).as_posix()
			if st.checkbox(relative_path, key=f"file_{relative_path}"):
				selected_files.append(file_path)

st.caption(f"{len(selected_files)} of {len(data_files)} files selected")

key_to_name: dict[int, str]= {
	123: 'Mr Smith',
	456: 'Ms Rose',
	654: 'Miss Jones',
	789: 'Mr Bean',
	876: 'Mr Thor',
	1001: 'Dr Amelia Jones'
}

st.subheader("Query")
options = [DocumentScope.ALL_CLIENTS.value, DocumentScope.INTERNAL.value] + list(key_to_name.keys())
scope_selection = st.selectbox("Client Reference", options=options)
if isinstance(scope_selection, int):
	st.caption(f"Client: {key_to_name[scope_selection]}")

with st.form("query_form"):
	query = st.text_input("Enter a question", placeholder="Ask something about the selected files")
	submitted = st.form_submit_button("Ask")

if submitted and query.strip():
	with st.spinner("Generating response...", show_time=True):
		st.session_state["query_result"] = answer_query(query, scope_selection)
	st.session_state["submitted_query"] = query

result = st.session_state.get("query_result")
if result:
	st.subheader("Answer")
	final_response = result["response"]
	st.markdown(
		final_response.get("response", "")
		if isinstance(final_response, dict)
		else final_response
	)

	prompt_tokens, completion_tokens = get_token_usage(result)
	st.subheader("Token usage")
	usage_columns = st.columns(4)
	usage_columns[0].metric("Total tokens", prompt_tokens + completion_tokens)
	usage_columns[1].metric("Prompt tokens", prompt_tokens)
	usage_columns[2].metric("Completion tokens", completion_tokens)
	usage_columns[3].metric("Response time", f"{result['response_time_seconds']:.2f} s")
	
	st.subheader("Citations")
	for i, match_data in enumerate(result["matches"], 1):
		match = match_data["match"]
		score = 1 - match["score"]
		with st.expander(f"{i}. {match['file_name']} — relevance: {score:.3f}"):
			with st.container(border=True):
				st.markdown(
					'<div style="background-color: grey; padding: 0.35rem 0.6rem; '
					'border-radius: 0.25rem;"><strong>Context</strong></div>',
					unsafe_allow_html=True,
				)
				st.markdown(match["context"])
			with st.container(border=True):
				st.markdown(
					'<div style="background-color: grey; padding: 0.35rem 0.6rem; '
					'border-radius: 0.25rem;"><strong>Extracted answer</strong></div>',
					unsafe_allow_html=True,
				)
				st.markdown(match_data["context_response"]["response"])
