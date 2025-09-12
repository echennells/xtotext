# Use official Ubuntu as base image (ARM64 compatible)
FROM --platform=linux/arm64 ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CC=clang
ENV CXX=clang++

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    clang \
    pkg-config \
    wget \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone whisper.cpp repository
RUN git clone https://github.com/ggerganov/whisper.cpp.git

# Change to whisper.cpp directory
WORKDIR /app/whisper.cpp

# Build whisper.cpp
# Note: Metal won't work in Docker container, but we'll build a CPU-optimized version
RUN make -j$(nproc)

# Download a base model for testing
RUN bash ./models/download-ggml-model.sh base

# Create a models directory and set permissions
RUN mkdir -p /app/models && \
    chmod 755 /app/models

# Copy models to accessible location
RUN cp models/*.bin /app/models/

# Create entrypoint script
RUN echo '#!/bin/bash\n\
echo "Whisper.cpp is ready!"\n\
echo "Available models in /app/models/"\n\
ls -la /app/models/\n\
echo ""\n\
echo "Usage examples:"\n\
echo "./main -m /app/models/ggml-base.bin -f your_audio_file.wav"\n\
echo ""\n\
echo "Starting interactive shell..."\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

# Expose any ports if needed (for server mode)
EXPOSE 8080
