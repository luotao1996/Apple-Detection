import time
import cv2
import argparse
from tensorpack import *
import tensorflow as tf
from inference import ResNetC4Model,detect_one_image
from config import finalize_configs, config as cfg
from viz import draw_fruits_box
from distance import get_distance, get_size

class Fruits:
    def __init__(self, fruit):
        self.box = fruit.box
        self.score = fruit.score
        self.cls = 'Apple' if fruit.class_id == 48 else 'Banana'
        distance = get_distance(fruit.box)
        self.distance = distance if fruit.class_id == 48 else distance * 1.25
        self.size = get_size(fruit.box)

# use models to detect
def process_detector_func(models, image_bgr):
    # Perform detection
    results = detect_one_image(image_bgr, models)
    # apple's id is 48,banana's id is 47
    fruits = [Fruits(r) for r in results if (r.class_id == 47 or r.class_id == 48)]

    # Draw fruits results
    image_disp = draw_fruits_box(image_bgr, fruits)
    return image_disp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help="This argument is the path to the input video file")
    parser.add_argument('--image', type=str, help="This argument is the path to the input image file")
    parser.add_argument('--cam', type=int, help='Specify which camera to detect.', default=0)
    parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py", nargs='+')

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    MODEL = ResNetC4Model()
    assert tf.test.is_gpu_available()
    assert args.load
    finalize_configs(is_training=False)

    cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

    pred = OfflinePredictor(PredictConfig(
        model=MODEL,
        session_init=get_model_loader(args.load),
        input_names=MODEL.get_inference_tensor_names()[0],
        output_names=MODEL.get_inference_tensor_names()[1]))

    if args.cam:
        # Read camera
        cap = cv2.VideoCapture(0)
    elif args.video:
        # Read video
        cap = cv2.VideoCapture(args.video)
    elif args.image:
        cap = cv2.VideoCapture(args.image)
    else:
        raise Exception("Either cam or video or image need to be specified as input")

    width, height = cap.get(3), cap.get(4)
    assert width, "Read image or video faild,please re-check your path"
    print((width, height))

    frame_count = 0
    while True:

        grabbed, image_bgr = cap.read()

        if not grabbed:
            break
        frame_count += 1
        t = time.time()
        img_to_show = process_detector_func(pred, image_bgr)
        print("Process frame {} takes {}s".format(frame_count, time.time() - t))

        if img_to_show is not None:
            cv2.imshow('video', img_to_show)
            if args.image:
                cv2.waitKey(0)
            elif args.video or args.cam:
                k = cv2.waitKey(1)
                if k == 27:  # Esc key to stop
                    break

'''
--image
/root/datasets/apple.jpg
--load
/root/datasets/COCO-R50C4-MaskRCNN-Standard.npz

'''
