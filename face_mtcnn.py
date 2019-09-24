import cv2
import os
import time
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

img_path = "photo"
if not os.path.exists(img_path):
    os.mkdir(img_path)

_CAMERA_WIDTH = 640  # 攝影機擷取影像寬度
_CAMERA_HEIGH = 480  # 攝影機擷取影像高度

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1)

# 設定擷取影像的尺寸大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAMERA_HEIGH)

while True:
    # Capture frame-by-frame
    __, frame = cap.read()

    # Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        k = 0
        for person in result:
            k += 1
            bounding_box = person['box']
            keypoints = person['keypoints']

            x = bounding_box[0]
            y = bounding_box[1]
            w = bounding_box[2]
            h = bounding_box[3]

            # 每隔 n 秒存一次照片
            n = 5
            t = int(time.time())
            padding = 20
            if t % n == 0:
                face_image = frame[y - padding:y + h + padding, x - padding:x + w + padding]
                storepath = img_path + "/img_item" + "_" + str(k) + "_" + str(t) + ".png"
                ## 写成图像
                cv2.imwrite(storepath, face_image)
                test = cv2.imread(storepath)
                if test is None:
                    os.remove(storepath)

            # 框出人臉
            cv2.rectangle(frame,
                          (x, y),
                          (x + w, y + h),
                          (0, 155, 255),
                          2)
            ##  画出标签
            cv2.circle(frame, (keypoints['left_eye']), 2, (0, 155, 255), 2)
            cv2.circle(frame, (keypoints['right_eye']), 2, (0, 155, 255), 2)
            cv2.circle(frame, (keypoints['nose']), 2, (0, 155, 255), 2)
            cv2.circle(frame, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
            cv2.circle(frame, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

    # display resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything's done, release capture
cap.release()
cv2.destroyAllWindows()

# mtcnn p-net \ r-net \ o-net
# https://towardsdatascience.com/face-detection-neural-network-structure-257b8f6f85d1