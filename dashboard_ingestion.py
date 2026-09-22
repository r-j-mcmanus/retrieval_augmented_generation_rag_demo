import os
import requests

import streamlit as st

# run with streamlit run dashboard_ingestion.py

VISIBILITY_CLIENT = 'Client'
VISIBILITY_INTERNAL = 'Internal'

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="RAG Ingestion", page_icon="⬆", layout="centered")
st.title("Index a document")
st.caption("Add a local document to the retrieval pipeline.")


@st.cache_data(ttl=30)
def get_existing_tags() -> list[str]:
	response = requests.get(f"{API_URL}/tags", timeout=10)
	response.raise_for_status()
	return response.json()


try:
	existing_tags = get_existing_tags()
except requests.RequestException as error:
	existing_tags = []
	st.warning(f"Existing tags are unavailable: {error}")

with st.form("ingestion_form"):
	filepath = st.text_input(
		"File path",
		placeholder=r"C:\documents\investment_policy.pdf",
		help="The file path must be accessible to the RAG API process.",
	)
	visibility = st.radio("Visibility", [VISIBILITY_INTERNAL, VISIBILITY_CLIENT], horizontal=True)

	client_reference = None
	if visibility == VISIBILITY_CLIENT:
		client_reference = st.text_input("Client reference", placeholder="C-001")

	selected_tags: list[str] = []
	if visibility == VISIBILITY_INTERNAL:
		selected_tags = st.multiselect(
			"Tags",
			options=existing_tags,
			help="Select existing tags or add new tags below.",
		)
		new_tags = st.text_input("New tags", placeholder="technical, process guide")
		selected_tags.extend(tag.strip() for tag in new_tags.split(",") if tag.strip())

	submitted = st.form_submit_button("Index document", type="primary")

if submitted:
	payload = {
		"file_path": filepath,
		"visibility": visibility.lower(),
		"client_reference": client_reference,
		"tags": sorted(set(selected_tags), key=str.casefold),
	}
	try:
		response = requests.post(f"{API_URL}/index", json=payload, timeout=300)
		response.raise_for_status()
	except requests.HTTPError as error:
		error_response = error.response
		detail = error_response.text if error_response is not None else str(error)
		if error_response is not None:
			try:
				detail = error_response.json().get("detail", detail)
			except ValueError:
				pass
		st.error(f"Indexing failed: {detail}")
	except requests.RequestException as error:
		st.error(f"Could not reach the RAG API: {error}")
	else:
		result = response.json()
		st.success(f"Indexed {result['filepath']} ({result['chunk_count']} chunks).")
		if result["tags"]:
			st.write("Tags: " + ", ".join(result["tags"]))
