import cv2
import change_image

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #만약 인식 실패했을 시, 전에 파악했던 얼굴 영역으로 대체
    px, py, pw, ph = 0, 0, 0, 0
    is_first = False

    while True:
        if cv2.waitKey(33) >= 0:
            break

        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray)
        #인식 실패시
        if isinstance(faces, tuple):
                change_image.change_mosaic(frame, ksize=20)
        #성공시
        else:
            for (x, y, w, h) in faces:
                change_image.change_mosaic(frame, (x, y, w, h), 20)
                is_first = True
                px = x
                py = y
                pw = w
                ph = h
        cv2.imshow('camera', frame)

    capture.release()
    cv2.destroyAllWindows()