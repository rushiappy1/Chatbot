FROM python:3.10-slim

# System deps (for pyodbc / MSSQL and build tools if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    unixodbc unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit

# Copy the rest of the application
COPY . .

# Streamlit config (can be overridden at runtime)
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

EXPOSE 7860

# Default command: run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
