# Use an official Python runtime as a parent image
FROM python:3.12.6-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies (including libpq-dev for PostgreSQL)
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Set environment variable for Alembic to locate the correct config
ENV ALEMBIC_CONFIG=/app/alembic.ini

ENV DEBUG_DATABASE_URL=sqlite:///./sql_app.db
ENV DATABASE_URL=postgresql://postgres:8vxH8bKEM0dQd0k25HZh@database-1.cna2i0i20lkp.af-south-1.rds.amazonaws.com:5432/
ENV DEBUG=0
ENV GROQ_API=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
ENV HUGGINGFACE_API_KEY=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
ENV MONGO_DB=mongodb+srv://oratilenailana:V8mZ2yCv60ENPSI3@cluster0.ofp6j.mongodb.net/embeddings?retryWrites=true&w=majority&appName=Cluster0
ENV DEBUG_MONGO_DB=mongodb://localhost:27017
ENV AWS_ACCESS_KEY_ID=AKIAXYKJQNNI6NIKXZKQ
ENV AWS_SECRET_ACCESS_KEY=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
ENV S3_BUCKET_NAME=articlabs-files

# Run Alembic migrations and then start the server
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8080"]
