import torch
import cv2
import os
import json

# TODO: 커스텀 모델 로드
model_path = os.path.abspath("./rps_model.onnx")
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)

# TODO: Label 로드
label_path = os.path.abspath("./rps_model.names.json")
with open(label_path, "r") as f:
    rabel_names = json.load(f)

# Video capture
cap = cv2.VideoCapture(0)

# Loop for camera frames
while True:
    # Read frame (BGR to RGB)
    ret, frame = cap.read()
    # break the loop on error
    if not ret:
        break

    # 추론 실행 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # TODO: 추론 전 입력 크기 보정 (640x640)
    rgb_frame = cv2.resize(rgb_frame, (640, 640))
    results = model(rgb_frame)

    # TODO: 카메라 입력의 크기(frame_h, frame_w)와 모델의 입력 크기(input_h, input_w) 구하기
    detections = results.xyxy[0]

    # 원본 프레임의 높이와 너비 가져오기
    frame_h, frame_w, _ = frame.shape

    # 스케일링 비율 계산
    x_scale = frame_w / 640
    y_scale = frame_h / 640

    # Boudning box 그리기
    for i, obj in enumerate(detections):
        # 인식결과를 표시하기 위한 좌표를 얻음
        x1, y1, x2, y2, _, cls = map(int, obj)
        conf = obj[4]

        # 스케일링된 좌표 계산
        x1_scaled = int(x1 * x_scale)
        y1_scaled = int(y1 * y_scale)
        x2_scaled = int(x2 * x_scale)
        y2_scaled = int(y2 * y_scale)

        # TODO: 인식된 정확도(confidence)와 클래스를 label로 구성
        label = rabel_names[f"{cls}"]

        # TODO: 출력 바운딩박스 크기 조절

        # OpenCV를 이용해서 해당 좌표에 사각형과 text를 출력
        cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1_scaled, y1_scaled - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"Object {i}: {label} at [{x1_scaled}, {y1_scaled}, {x2_scaled}, {y2_scaled}]")

    # 화면 표시
    cv2.imshow("YOLOv5", frame)

    # 종료를 위한 key 처리
    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()