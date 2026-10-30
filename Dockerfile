FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir fastapi uvicorn
COPY model.py data.py train.py serve.py ./
RUN python train.py && useradd --create-home triage
USER triage
EXPOSE 8001
CMD ["uvicorn", "serve:create_app", "--factory", "--host", "0.0.0.0", "--port", "8001", "--no-access-log"]
