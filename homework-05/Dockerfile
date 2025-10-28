FROM agrigorev/zoomcamp-model:2025

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN pip install --no-cache-dir -e .

# Copy the FastAPI application
COPY main.py ./

COPY pipeline_v1.bin ./

# Expose the port
EXPOSE 9000

# Run the application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]