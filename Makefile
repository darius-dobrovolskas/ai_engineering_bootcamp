run-streamlit:
	streamlit run src/chatbot_ui/streamlit_app.py

build-docker-streamlit:
	docker build -t streamlit_app:latest .

run-docker-streamlit:
	docker run -v ${PWD}/.env:/app/.env -p 8501:8501 streamlit_app:latest

run-docker-compose:
	docker compose up --build

run-evals:
	poetry install
	$env:PYTHONPATH = ".\src"
	>> poetry run python -m evals.eval_retriever