FROM python:3.12-slim

WORKDIR /app

# Enable bytcode compilation and Python optimization
ENV PYTHONOPTIMIZE=1

# Set Python path to include the src directory for imports
ENV PYTHON="/app/src"

# Install Poetry
RUN pip install --no-cache-dir poetry

# Prevent Poetry from creating a virtualenv
ENV POETRY_VIRTUALENVS_CREATE=false

# Copy only dependency files first for better layer cashing
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-interaction --no-root

# Copy application code
COPY src/chatbot_ui ./src/chatbot_ui

# Set PYTHONPATH so Python can find modules under /app/src
ENV PYTHONPATH="/app/src"

# Pre-compile Python files to bytcode
RUN python -m compileall ./src/chatbot_ui/

# Create non-root user and set permissions
RUN addgroup --system app && \
    adduser --system --ingroup app --home /home/app app && \
    chown -R app:app /app

# Set HOME environment variable for the app user
ENV HOME=/home/app

# Swith to non-root user
USER app

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "src/chatbot_ui/streamlit_app.py", "--server.address=0.0.0.0"]