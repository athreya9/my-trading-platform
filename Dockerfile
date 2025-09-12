# Use an official Python runtime as a parent image.
# Using 'slim' reduces the final image size.
FROM python:3.12-slim

# Set the working directory in the container to /app
WORKDIR /app

# Install system dependencies that might be needed by Python packages.
# This helps prevent build failures for packages with C extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    # Add chromium and chromedriver for Selenium
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt.
# --no-cache-dir reduces image size by not storing the pip cache.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container at /app
COPY . .

# Make port 8080 available to the world outside this container.
# Cloud Run will automatically map this to its external port.
EXPOSE 8080

# Define the command to run the app using gunicorn, a production-ready WSGI server.
# The PORT environment variable is automatically set by Cloud Run.
# --timeout 0 disables the worker timeout, which is crucial for long-running tasks.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 0 app:app"]