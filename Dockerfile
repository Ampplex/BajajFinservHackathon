# Use a slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies file
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app code
COPY . .

# Expose default Streamlit port
EXPOSE 8080

# Run Streamlit on port 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false"]