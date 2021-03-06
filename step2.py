import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2

# Step 2: Replace background behind subject with an image
#
# In this script, we use the raw video from step 1, a still background image,
# and the mask generated by the segmentation process, to layer the new
# background behind the subject.
#
# There is no built-in concept of layers, so we have to add the two layers
# together mathematically.
#
# The mask is a single channel (not RGB) with a 1 in every pixel where the
# subject is, and a 0 zero everywhere else. This means that if we multiply the
# webcam feed by the mask, we will end up with the original pixels under the
# mask and black everywhere else.
#
# Similarly, if we multiply our new background by (1 - mask), we will get black
# everywhere BUT where the subject is. This means that there is no overlap in
# non-black pixels between the subject and background, so we can then just
# add them together.
#
# This is exactly the same process that was used for green screens in early
# movies (and is still what happens behind the scenes any time a computer
# overlays two images).


def main():
    # Initialise webcam video
    height = 480
    width = 640
    capture = cv2.VideoCapture(0)
    capture.set(3, width)
    capture.set(4, height)

    # Create an instance of the Resnet50 model
    ins = instanceSegmentation()
    ins.load_model("pointrend_resnet50.pkl")

    # Load replacement background image
    background_image = cv2.imread("background.jpg")
    background_image = background_image.astype("uint8")

    # Select the resnet50 classes you want to segment by
    target_classes = ins.select_target_classes(person=True)

    # Run this every frame
    while True:
        ret, frame = capture.read()

        # Segment people from image
        segment_mask, output = ins.segmentFrame(
            frame.copy(),
            segment_target_classes=target_classes,
            extract_segmented_objects=False)

        # Extract first mask from segmented masks and create inverse
        mask = segment_mask["masks"].astype("uint8")

        mask.resize((mask.shape[0], mask.shape[1], 1), refcheck=False)

        mask_inverse = 1 - mask
        mask_inverse = mask_inverse.astype("uint8")

        # Composite background with foreground using mask
        bg = cv2.bitwise_and(background_image,
                             background_image,
                             mask=mask_inverse)
        fg = cv2.bitwise_and(frame, frame, mask=mask)

        composite = cv2.add(bg, fg)

        # Display results in windows
        cv2.imshow("composite", composite)

        # If the user presses the q key, close the window
        key = cv2.waitKey(25)
        if key & 0xff == ord('q'):
            break


if __name__ == "__main__":
    main()
