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
    change_fn = [change_image.change_mosaic, change_image.change_canny, change_image.change_max]
    fn_num = 0

    while True:
        key = cv2.waitKey(33)
        if key == ord('q') or key == ord('Q'):
            fn_num = 0
        elif key == ord('w') or key == ord('W'):
            fn_num = 1
        elif key == ord('e') or key == ord('E'):
            fn_num = 2
        elif key >= 0:
            break

        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray)
        #인식 실패시
        if isinstance(faces, tuple):
                change_fn[fn_num](frame)
        #성공시
        else:
            for (x, y, w, h) in faces:
                change_fn[fn_num](frame, (x, y, w, h), 20)
                is_first = True
                px = x
                py = y
                pw = w
                ph = h
        cv2.imshow('camera', frame)

    capture.release()
    cv2.destroyAllWindows()