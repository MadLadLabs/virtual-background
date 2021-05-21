FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libv4l-dev

RUN pip install \
    tf_bodypix \
    opencv-python \
    tfjs_graph_converter \
    numpy \
    pyfakewebcam

COPY src /app

WORKDIR /app

CMD python app.py