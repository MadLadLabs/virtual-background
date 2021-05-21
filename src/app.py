import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import cv2
import numpy as np
import pyfakewebcam

bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))
# bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_RESNET50_FLOAT_STRIDE_16))

# get vid cap device
cap = cv2.VideoCapture('/dev/video0') 
height, width = 720, 1280
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 30)

fake = pyfakewebcam.FakeWebcam('/dev/video20', width, height)

def shift_img(img, dx, dy):
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy>0:
        img[:dy, :] = 0
    elif dy<0:
        img[dy:, :] = 0
    if dx>0:
        img[:, :dx] = 0
    elif dx<0:
        img[:, dx:] = 0
    return img

# loop through frame
while cap.isOpened(): 
    ret, frame = cap.read()
    holo = cv2.applyColorMap(frame, cv2.COLORMAP_WINTER)
    bandLength, bandGap = 2, 3
    for y in range(holo.shape[0]):
        if y % (bandLength+bandGap) < bandLength:
            holo[y,:,:] = holo[y,:,:] * np.random.uniform(0.1, 0.3)

    # the first one is roughly: holo * 0.2 + shifted_holo * 0.8 + 0
    holo2 = cv2.addWeighted(holo, 0.2, shift_img(holo.copy(), 5, 5), 0.8, 0)
    holo2 = cv2.addWeighted(holo2, 0.4, shift_img(holo.copy(), -5, -5), 0.6, 0)

    holo_done = cv2.addWeighted(frame, 0.6, holo2, 0.4, 0)
    
    green_frame = np.full(frame.shape, [0,255,0], dtype=np.uint8)

    # BodyPix Detections
    result = bodypix_model.predict_single(frame)
    mask = result.get_mask(threshold=0.5).numpy().astype(np.uint8) * 255
    mask_inverted = cv2.bitwise_not(mask)
    masked_image = cv2.bitwise_and(holo_done, holo_done, mask=mask)
    masked_green = cv2.bitwise_and(green_frame, green_frame, mask=mask_inverted)
    
    result = cv2.add(masked_image, masked_green)

    fake.schedule_frame(result)

    # # Show result to user on desktop
    # cv2.imshow('BodyPix', result)
    
    # # Break loop outcome 
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break

cap.release() # Releases webcam or capture device
cv2.destroyAllWindows() # Closes imshow frames