
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN pip install lightning==2.4.*
RUN pip install pystoi \
    pesq \
    librosa \ 
    scipy \ 
    soundfile \ 
    python_speech_features \
    onnxruntime \
    parse \
    pandas \
    loguru \
    matplotlib 





