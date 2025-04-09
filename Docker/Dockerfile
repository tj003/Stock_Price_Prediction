FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

COPY requirements.txt .

# Install dependencies (using --no-cache-dir to reduce image size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run the app
CMD ["python", "app.py"]
