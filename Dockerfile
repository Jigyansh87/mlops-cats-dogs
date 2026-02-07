FROM python:3.10

WORKDIR /app

# Copy inference code
COPY app.py .

# Copy trained model
COPY models ./models

EXPOSE 8000

# Run FastAPI using Python entrypoint
CMD ["python", "app.py"]
