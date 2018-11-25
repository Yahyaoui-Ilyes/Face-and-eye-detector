#######################
# Author : Ilyes Yahyaoui     #
#           Contact:                  #
# Yahyaouilyes@gmail.com #
#######################
import cv2
face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Loading the frontal face haarcascade
eye_cascade= cv2.CascadeClassifier('haarcascade_eye.xml') # Loading the frontal face haarcascade
capture= cv2.VideoCapture(0) # Creating a capture object

while True:
    ret , img = capture.read() # Capture frame-by-frame
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Converting into GRAYSCALE
    faces = face_cascade.detectMultiScale(gray, 1.3,  5)

    for(x,y,w,h) in faces:
        cv2.rectangle   (  img,    (x,y),   (x+w , y+h),      (0,0,255),     2   )# Setting the face rectangle
        roi_gray    =   gray[y:y+h,x:x+w]
        roi_color   = img[y:y+h,x:x+w]
        
        eyes= eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,eh,ew) in eyes:
            cv2.rectangle(  roi_color,  (ex,ey),    (ex+ew, ey+eh),     (255,0,0),  2)# Setting the eyes rectangles

    cv2.imshow('img',img)# Display the resulting frame

    k = cv2.waitKey(30) & 0xff # Setting the exit key to escape
    if k==27:
        break
    
capture.release() #Releasing Capture
cv2.destroyAllWindows()#Closing the window
