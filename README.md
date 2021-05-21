# virtual-background

sudo apt install v4l2loopback-dkms v4l-utils
sudo modprobe v4l2loopback exclusive_caps=1 video_nr=10 card_label="OBS Stream"


```
docker run -e DISPLAY=$DISPLAY \
    --device=/dev/video0:/dev/video0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --volume path/to/workspace:/tf-workspace \
    --gpus all \
    spiridonovpolytechnic/virtual-background:latest
```

docker run -e DISPLAY=$DISPLAY \
    --device=/dev/video0:/dev/video0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --gpus all \
    vb