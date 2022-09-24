import cv2 as cv
import face_recognition

webcam = cv.VideoCapture(0)

face_locations = []

trained_data = cv.CascadeClassifier('frontal-face-data.xml')





while True:
    check, frame = webcam.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)

    cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2) 

    cv.imshow('video', frame)

    key = cv.waitKey(1)
    if key == ord('e'):
        break

webcam.release()
cv.destroyAllWindows()