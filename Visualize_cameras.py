import cv2

# Create VideoCapture objects for multiple cameras
camera_0 = cv2.VideoCapture(0)
camera_1 = cv2.VideoCapture(2)

while True:
    # Read frames from cameras
    ret_0, frame_0 = camera_0.read()
    ret_1, frame_1 = camera_1.read()

    # Display frames, if successfully read
    if ret_0:
        cv2.imshow('Camera 0', frame_0)
    if ret_1:
        cv2.imshow('Camera 1', frame_1)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture objects and destroy windows
camera_0.release()
camera_1.release()
cv2.destroyAllWindows()