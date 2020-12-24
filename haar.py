from io import BytesIO
from PIL import Image as Img
import cv2 as cv

import matplotlib.image as mpimg

if __name__ == '__main__':

    imagefile = mpimg.imread(
        'C:\\Users\\alyona\\Documents\\cloud\\cloud_app\\FV000046.jpg')

    gray = cv.cvtColor(imagefile, cv.COLOR_BGR2GRAY)

    cascade_path = r'' + cv.data.haarcascades + "haarcascade_frontalface_default.xml"

    face_cascade = cv.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    print("[INFO] Found {0} Faces.".format(len(faces)))

    for (x, y, w, h) in faces:
        cv.rectangle(imagefile, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = imagefile[y:y + h, x:x + w]
        print("[INFO] Object found. Saving locally.")
        roi_to_save = cv.cvtColor(roi_color, cv.COLOR_BGR2RGB)
        cv.imwrite(str(w) + '_' + str(h) + '_faces.jpg', roi_to_save)

        image = cv.cvtColor(roi_color, cv.COLOR_BGR2RGB)
        img_crop_pil = Img.fromarray(image)
        byte_io = BytesIO()
        img_crop_pil.save(byte_io, 'jpeg')
        jpg_buffer = byte_io.getvalue()
        byte_io.seek(0)

    imagefile_to_save = cv.cvtColor(imagefile, cv.COLOR_BGR2RGB)
    status = cv.imwrite('faces_detected.jpg', imagefile_to_save)
    print("[INFO] Image faces_detected.jpg written to filesystem: ", status)
