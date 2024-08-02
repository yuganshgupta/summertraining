import cv2 

# Initialize the camera
cam = cv2.VideoCapture(0)

# Load the face detector
path = 'C:\\ProgramData\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(path)

while True:
    # Read a frame from the camera
    val, im = cam.read()

    # Check if the frame was read properly
    if not val:
        print("Frame not read properly")
        break

    # Resize the frame
    im_new = cv2.resize(im, (512,512))

    # Convert the frame to grayscale
    gray_im = cv2.cvtColor(im_new, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray_im, scaleFactor=1.1, minNeighbors=10)

    # Draw rectangles around the faces
    for (dx, dy, w, h) in faces:
        cv2.rectangle(im_new, (dx, dy), (dx+w, dy+h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('camera live feed', im_new)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cam.release()
cv2.destroyAllWindows()
