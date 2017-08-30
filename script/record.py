import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

toggle = False
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if toggle:
        out.write(frame)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        print('Record mode:', toggle)
        toggle = not toggle

cap.release()
out.release()
cv2.destroyAllWindows()
