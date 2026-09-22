@echo off
setlocal

echo Starting RAG system...

start "Streamlit Dashboard" cmd /k "streamlit run dashboard.py"
start "Streamlit Ingestion" cmd /k "streamlit run dashboard_ingestion.py"
start "RAG API" cmd /k "uvicorn api_rag:app --host 127.0.0.1 --port 8000"
start "LLM API" cmd /k "uvicorn api_llm:app --host 127.0.0.1 --port 8001"

echo.
echo All services started.
echo.
echo Dashboard:        http://localhost:8501
echo Ingestion:        http://localhost:8502
echo RAG API:          http://127.0.0.1:8000
echo LLM API:          http://127.0.0.1:8001
echo.

endlocal
