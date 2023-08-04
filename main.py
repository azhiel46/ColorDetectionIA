import cv2
import imutils
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Amarillo --- #
    amarillo_osc = np.array([25, 70, 120])
    amarillo_cla = np.array([30, 255, 255])

    # --- Rojo --- #
    rojo_osc = np.array([0, 50, 120])
    rojo_cla = np.array([10, 255, 255])

    # --- Verde --- #
    verde_osc = np.array([40, 70, 80])
    verde_cla = np.array([70, 255, 255])

    # --- Azul --- #
    azul_osc = np.array([90, 60, 0])
    azul_cla = np.array([121, 255, 255])

    mask1 = cv2.inRange(hsv, amarillo_osc, amarillo_cla)
    mask2 = cv2.inRange(hsv, rojo_osc, rojo_cla)
    mask3 = cv2.inRange(hsv, verde_osc, verde_cla)
    mask4 = cv2.inRange(hsv, azul_osc, azul_cla)

    cnts1 = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)

    cnts2 = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)

    cnts3 = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts3 = imutils.grab_contours(cnts3)

    cnts4 = cv2.findContours(mask4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts4 = imutils.grab_contours(cnts4)

    for c in cnts1:
        area1 = cv2.contourArea(c)
        if area1>5000:
            cv2.drawContours(frame, [c], -1, (30, 255, 255), 3)
            M = cv2.moments(c)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Amarillo", (cx-20, cy-20), cv2.FONT_ITALIC, 2, (255, 255, 255), 2)

    for c in cnts2:
        area2 = cv2.contourArea(c)
        if area2>5000:
            cv2.drawContours(frame, [c], -1, (0, 0, 255), 3)
            M = cv2.moments(c)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Rojo", (cx-20, cy-20), cv2.FONT_ITALIC, 2, (255, 255, 255), 2)

    for c in cnts3:
        area3 = cv2.contourArea(c)
        if area3>5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            M = cv2.moments(c)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Verde", (cx-20, cy-20), cv2.FONT_ITALIC, 2, (255, 255, 255), 2)

    for c in cnts4:
        area4 = cv2.contourArea(c)
        if area4>5000:
            cv2.drawContours(frame, [c], -1, (255, 0, 0), 3)
            M = cv2.moments(c)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Azul", (cx-20, cy-20), cv2.FONT_ITALIC, 2, (255, 255, 255), 2)

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()