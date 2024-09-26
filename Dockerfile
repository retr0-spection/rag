# Use an official Python runtime as a parent image
FROM python:3.12.6-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Set environment variable for Alembic to locate the correct config
ENV ALEMBIC_CONFIG=/app/alembic.ini

ENV DATABASE_URL=postgresql://postgres:8vxH8bKEM0dQd0k25HZh@database-1.cna2i0i20lkp.af-south-1.rds.amazonaws.com:5432/

# Run Alembic migrations and then start the server
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8080"]
