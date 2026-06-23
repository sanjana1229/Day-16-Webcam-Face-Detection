from facedetector import FaceDetector
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True,
    help="path to face cascade")
ap.add_argument("-v", "--video",
    help="path to optional video file")

args = vars(ap.parse_args())

fd = FaceDetector(args["face"])

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    grabbed, frame = camera.read()

    if args.get("video") and not grabbed:
        break

    if not grabbed:
        print("Cannot access webcam")
        break

    frame = cv2.resize(frame, (300, int(frame.shape[0] * 300 / frame.shape[1])))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceRects = fd.detect(gray)

    for (x, y, w, h) in faceRects:
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                     (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()