import datetime

import cv2
from Action_Recognition import PreProcessing

cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cv2.ocl.setUseOpenCL(False)
action = PreProcessing()
while cam:
    _, frame = cam.read()
    start = datetime.datetime.now()
    action.action_recognition(frame)
    time = datetime.datetime.now() - start
    if time.total_seconds() > 1:
        pass
        print(action.result())
        print("Inference Time:")
        print(time.total_seconds())
    cv2.imshow("", action.draw_box_on_frame(frame))
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
