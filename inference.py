import cv2  # OpenCV for video and image processing
from utilis import label_detection, YOLO_Detection
from ultralytics import YOLO


def load_yolo_model(path, device):
    """Load the YOLO model and configure it."""
    model = YOLO(path)
    model.to(device)
    model.nms = 0.7
    print(f"Model classes: {model.names}")
    return model


def run_detection(model_weights, source):
    """
    Main function to run parking lot detection. Loads video streams and pre-defined 
    parking spot positions, performs detection, and displays the occupancy status.

    Args:
    json_file (str): Path to the JSON file containing video URLs and index data.
    """

    # Open the video stream using the specified URL
    cap = cv2.VideoCapture(source)

    model = load_yolo_model(path = model_weights, device="cuda")

    # Check if the video stream was successfully opened
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return  # Exit the function if the video stream fails to open

    # Loop to process frames from the video stream
    while True:
        ret, frame = cap.read()  # Read a frame from the video stream
        if not ret:  # If no frame is retrieved (end of stream or error), exit the loop
            break

        boxes, classes, names, confidences = YOLO_Detection(model, frame, conf=0.4)

        for box, cls in zip(boxes, classes):
            if int(cls) == 1:
                label_detection(frame=frame, text=f"{names[int(cls)]}", tbox_color=(255, 144, 30),
                                left=box[0], top=box[1], bottom=box[2], right=box[3])
            else:
                label_detection(frame=frame, text=f"{names[int(cls)]}", tbox_color=(0, 0, 230),
                                left=box[0], top=box[1], bottom=box[2], right=box[3])

        # Resize the frame for display (optional, based on your window size)
        frame_resized = cv2.resize(frame, (920, 640))  # Resize the frame to 920x640 for better display
        cv2.imshow("Frame", frame_resized)  # Show the processed frame in a window

        # Wait for a key press (or continue if no key is pressed)
        if cv2.waitKey(1) == ord("q"):  # Modify this if you want real-time display instead of waiting for a key press
            break

    # Clean up: release the video stream and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Uncomment the line below to run the detection function when needed
run_detection(model_weights="runs/detect/train4/weights/best.pt", source="car_data/IMG_6328.MOV")