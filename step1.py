import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2

# Step 1: Read input from webcam and segment
#
# In this script, we just read the live video feed from the webcam and then
# use a pretrained Resnet50 neural net to segment (separate) the subject from
# the background.
#
# Other object classes as well as "person" can be used by editing line 25.


def main():
    # Initialise the video feed
    height = 480
    width = 640
    capture = cv2.VideoCapture(0)
    capture.set(3, width)
    capture.set(4, height)

    # Create an instance of the Resnet50 model
    ins = instanceSegmentation()
    ins.load_model("pointrend_resnet50.pkl")

    # Select the resnet50 classes you want to segment by
    target_classes = ins.select_target_classes(person=True)

    while True:
        ret, frame = capture.read()

        # Segment people from image
        segment_mask, output = ins.segmentFrame(frame,
                                                segment_target_classes=target_classes,
                                                extract_segmented_objects=False,
                                                show_bboxes=True)

        # Show the segmented video in a window
        cv2.imshow('frame', frame)

        # If the user presses the q key, close the window.
        key = cv2.waitKey(25)
        if key & 0xff == ord('q'):
            break


if __name__ == "__main__":
    main()
