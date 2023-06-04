import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('PoseVideos/5.mp4')
pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)
    #print(results) -> this will print the differrent landmarks detected as classes

    #To get the information of each detected class
    #print(results.pose_landmarks)

    #Drawing the landmarks if detected
    if results.pose_landmarks:
        #mpDraw.draw_landmarks(img, results.pose_landmarks) #To show the landmarks points alone

        #To show the landmarks and coneect them with lines
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        #To store the landmarks as a list to ease the access of specific landmarks
        for id, lm in enumerate(results.pose_landmarks.landmark):
            print(id, lm) #-> it can be seen that the x and y values are in ratios to the pixel values

            #To the get actual pixel values of the landmarks
            h, w, c = img.shape

            cx = int(lm.x*w)
            cy = int(lm.y*h)
            #To recheck the values by drawing circles at these locations
            cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv2.imshow("Image", img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()