import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2
import numpy as np #machine learning library to deal with huge data
import torch
from torch.autograd import Variable

from net import Net #neural network
from utils import StyleLoader


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

    # Load foreground image
    background_image = cv2.imread("flower.jpg")
    background_image = background_image.astype("uint8")
    # Load background image
    background_imagee = cv2.imread("background.jpg")
    background_imagee = background_imagee.astype("uint8")
    # Select the resnet50 classes you want to segment by
    target_classes = ins.select_target_classes(person=True)

    # Initialize NST
    style_model = Net(ngf=128)#next cell
    model_dict = torch.load("models/21styles.model")
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)
    style_model.eval()
    style_loader = StyleLoader("images/21styles/", 512, False)

    style_idx = 0

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

        mask_inverse = mask

        # Composite background with foreground using mask
        bg = cv2.bitwise_and(background_image,
                             background_image,
                             mask=mask_inverse)
        bgr = cv2.bitwise_and(background_imagee,
                         background_imagee,
                         mask=1-mask_inverse)
        fg = cv2.bitwise_and(frame, frame, mask=mask)
        composite = cv2.add(bg, bgr, fg)

        # Prepare for NST
        nst_image = np.array(composite).transpose(2, 0, 1)
        # Load NST Style
        style_v = style_loader.get(style_idx)
        style_v = Variable(style_v.data)
        style_model.setTarget(style_v)

        # Apply style to image
        nst_image = torch.Tensor(nst_image).unsqueeze(0).float()
        nst_image = Variable(nst_image)
        nst_image = style_model(nst_image)

        styled_nst_image = style_v.data.numpy()
        nst_image = nst_image.clamp(0, 255).data[0].numpy()
        styled_nst_image = np.squeeze(styled_nst_image)
        nst_image = nst_image.transpose(1, 2, 0).astype('uint8')
        styled_nst_image = styled_nst_image.transpose(1, 2, 0).astype('uint8')

        # Display results in windows
        cv2.imshow("composite", nst_image)
        cv2.imshow("frame", frame)

        # If the user presses the q key, close the window
        # If they press the n key, cycle to the next style
        key = cv2.waitKey(25)
        if key & 0xff == ord('q'):
            break
        if key & 0xff == ord('n'):
            style_idx += 1


if __name__ == "__main__":
    main()
