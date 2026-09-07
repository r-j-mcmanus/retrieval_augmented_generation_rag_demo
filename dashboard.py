from pathlib import Path
import requests

import streamlit as st

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

def answer_query(query: str):
	response = requests.post(
		"http://127.0.0.1:8000/query",
		json={"query": query},
	)
	response.raise_for_status()
	return response.json()


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

st.subheader("Query")
with st.form("query_form"):
	query = st.text_input("Enter a question", placeholder="Ask something about the selected files")
	submitted = st.form_submit_button("Ask")

if submitted and query.strip():
	st.session_state["query_result"] = answer_query(query)
	st.session_state["submitted_query"] = query

result = st.session_state.get("query_result")
if result:
	st.subheader("Answer")
	st.markdown(result["response"])
	
	st.subheader("Citations")
	for i, match in enumerate(result["matches"], 1):
		score = 1 - match["score"]
		with st.expander(f"{i}. {match['source']} — relevance: {score:.3f}"):
			st.markdown("**Context**")
			st.markdown('...' + match["context"] + '...')
