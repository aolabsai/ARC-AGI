FROM python:3.12-slim

WORKDIR /app

# Install dependencies for building and compiling packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    openssh-client \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Setup SSH known hosts for secure GitHub connections
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan -H github.com >> ~/.ssh/known_hosts

# Mount the SSH agent to install from private repos securely
RUN --mount=type=ssh \
    pip install git+ssh://git@github.com/aolabsai/ao_core.git \
                git+git://github.com/aolabsai/ao_arch.git 

# Copy project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask app port
EXPOSE 5000


ENTRYPOINT ["python", "app.py"]


HEALTHCHECK CMD curl --fail http://localhost:5000 || exit 1
