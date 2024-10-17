# ao_app/Dockerfile
#
# Dev Notes: 
#    - compared to aolabsai/MNIST_streamlit and aolabsai/Recommender, this Dockerfile is different to support Flask (note the default Flask port:5000)

# First, build this container by running the 2 commands below in a Git Bash terminal:
# $ export DOCKER_BUILDKIT=1
# $ docker build --secret id=env,src=.env -t "ao_app" .

# Then, run the container with this command:
# $ docker run -p 5000:5000 "ao_app"

# You can then access your app at: http://localhost:5000/


FROM python:3.12-slim

# Create a directory for the app in the container
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install git and other necessary packages
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install python-dotenv

# Copy the app code (including templates and static files)
COPY . .

# Install AO modules, ao_core, and ao_arch
RUN --mount=type=secret,id=env,target=/app/.env \
    export $(grep -v '^#' .env | xargs) && \
    pip install git+https://${ao_github_PAT}@github.com/aolabsai/ao_core.git
RUN pip install git+https://github.com/aolabsai/ao_arch.git

# Expose Flask default port
EXPOSE 5000

# Healthcheck endpoint for Flask
HEALTHCHECK CMD curl --fail http://localhost:5000/_stcore/health || exit 1

# Start the Flask application
# ENTRYPOINT ["python", "app.py"]
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5000", "-w", "4", "wsgi:app"]
