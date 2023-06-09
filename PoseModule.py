import cv2
import mediapipe as mp
import time

#Creating Class, which will be able to create objects and hv methods to detect pose and find these points for us
class poseDetector():

    def __init__(self,
                 static_image_mode = False,
                 model_complexity = 1,
                 smooth_landmarks = True,
                 enable_segmentation = False,
                 smooth_segmentation = True,
                 min_detection_confidence = 0.5,
                 min_tracking_confidence = 0.5):

        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks,
                                     self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence,
                                     self.min_tracking_confidence)

    #Creating Method to find Pose
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img


    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx = int(lm.x*w)
                cy = int(lm.y*h)

                #Appending the values
                lmList.append([id,cx,cy])

                if draw:
                    #To recheck the values by drawing circles at these locations
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)

        return lmList



#Making the code a module
def main():
    cap = cv2.VideoCapture('PoseVideos/1.mp4')
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        #print(lmList) #-> to get all the list
        if len(lmList) != 0:
            print(lmList[14]) #-> to get a particular id in list (14-> left elbow)
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


cv2.destroyAllWindows()

if __name__ == "__main__":
    main()