run-streamlit:
	streamlit run src/chatbot_ui/streamlit_app.py

build-docker-streamlit:
	docker build -t streamlit_app:latest .

run-docker-streamlit:
	docker run -v ${PWD}/.env:/app/.env -p 8501:8501 streamlit_app:latest

run-docker-qdrant:
	docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:qdrant/storage:z" qdrant/qdrant

run-docker-compose:
	docker compose up --build

run-evals:
	poetry install
	$env:PYTHONPATH = ".\src"
	>> poetry run python -m evals.eval_retriever

run-evals-coordinator:
    poetry install
	$env:PYTHONPATH = (Join-Path (Get-Location) 'src') + ';' + ($env:PYTHONPATH)
	poetry run python -m evals.eval_coordinator_agent
