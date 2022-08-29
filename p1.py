import cv2

cap = cv2.VideoCapture(1)
print(cap.isOpened())
while(cap.isOpened()):
    ret, img = cap.read()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("img",img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()