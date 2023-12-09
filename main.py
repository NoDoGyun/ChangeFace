import cv2
import change_image
import numpy as np

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #전체화면
    is_full = False
    change_fn = [change_image.change_mosaic, change_image.change_canny, change_image.change_max]
    fn_num = 0
    #얼굴 영역 분류 위한 배열
    face_input = []
    face_target = []
    count = 0
    
    #얼굴 영역 샘플
    while cv2.waitKey(33) < 0:
        if count > 2000:
            break
        count += 33
        _, frame = capture.read()
        faces = cascade.detectMultiScale(frame)
        if not isinstance(faces, tuple):
            for (x, y, w, h) in faces:
                face_input.append([x, y, w, h])
                cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)
        text = "Getting Data: " + str((int)(count / 2000 * 100)) + "%"
        cv2.putText(frame, text, (50, 50), fontScale=1, color=(0, 0, 0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=3)
        cv2.imshow('face', frame)
    cv2.destroyAllWindows()

    #얼굴 중앙값 찾기
    face_input = np.array(face_input)
    input2 = face_input[:, 2]
    median = np.median(face_input)
    cor_face = None
    for i in face_input:
        if i[2] == median:
            cor_face = i
            break
    #얼굴으로 판단할 지 정하는 수치
    face_point = 50

    #얼굴 변환
    while True:
        key = cv2.waitKey(33)
        #원하는 모드, 전체화면 설정
        #모자이크
        if key == ord('q') or key == ord('Q'):
            fn_num = 0
        #윤곽선
        elif key == ord('w') or key == ord('W'):
            fn_num = 1
        #최댓값 필터
        elif key == ord('e') or key == ord('E'):
            fn_num = 2
        #전체화면 해제
        elif key == ord('d') or key == ord('D'):
            is_full = False
        #전체화면
        elif key == ord('f') or key == ord('F'):
            is_full = True
        #그 외 키 입력시 종료
        elif key >= 0:
            break

        ret, frame = capture.read()
        #전체화면 변경
        if is_full:
            change_fn[fn_num](frame)
        #얼굴만 변경
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray)
            # 인식 실패시
            if isinstance(faces, tuple):
                change_fn[fn_num](frame, (cor_face[0], cor_face[1], cor_face[2], cor_face[3]))
            # 성공 시
            else:
                for (x, y, w, h) in faces:
                    #성공한 영역이 얼굴 같지 않을 때
                    if abs(w - median) > face_point:
                        change_fn[fn_num](frame, (cor_face[0], cor_face[1], cor_face[2], cor_face[3]))
                    #성공시 
                    else:
                        change_fn[fn_num](frame, (x, y, w, h))
                        cor_face = [x, y, w, h]

        cv2.imshow('camera', frame)
    
    #종료
    capture.release()
    cv2.destroyAllWindows()
