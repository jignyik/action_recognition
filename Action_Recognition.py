
import numpy as np

import cv2
import torch

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from slowfast.datasets import cv2_transform
from slowfast.datasets.cv2_transform import scale

import onnxruntime


class PreProcessing:
    def __init__(self):
        self.device = "cpu"
        self.saved_frames = []
        self.frame_length = 32
        self.sample_rate = 2
        self.slowfast_alpha = 4
        self.sqn_length = self.frame_length*self.sample_rate
        self.predictor = self.load_detectron_config()
        self.session = self.load_onnx_session()
        self.class_names = self.load_class_names()
        self.height, self.width, self.channel = None, None, None
        self.new_boxes = np.array([])
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
        self.crop_size = 256
        self.pred_labels = None
        self.palette = np.random.randint(64, 128, (len(self.class_names), 3)).tolist()
        self.preds = np.array([])
        self.input_name = self.session.get_inputs()[0].name
        self.input_name2 = self.session.get_inputs()[1].name
        self.input_name3 = self.session.get_inputs()[2].name
        self.output_name = self.session.get_outputs()[0].name
        self.midframe_resized = np.array([])
        self.midframe = np.array([])

    @staticmethod
    def load_onnx_session():
        model = onnxruntime.InferenceSession(r"model_dynamic.onnx")
        return model

    @staticmethod
    def load_class_names():
        path_to_csv = "ava.names"
        with open(path_to_csv) as f:
            labels = f.read().split('\n')[:-1]
        return labels

    @staticmethod
    def load_detectron_config():
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        dummy = np.zeros((480,640,3))
        dummy = predictor(dummy)
        return predictor

    def preprocess(self, frames):
        # The mean value of the video raw pixels across the R G B channels.
        mean = [0.45, 0.45, 0.45]
        # The std value of the video raw pixels across the R G B channels.
        std = [0.225, 0.225, 0.225]
        inputs = torch.as_tensor(np.array(frames)).float()

        inputs = inputs / 255.0
        # Perform color normalization.
        inputs = inputs - torch.tensor(mean)
        inputs = inputs / torch.tensor(std)
        # T H W C -> C T H W.
        inputs = inputs.permute(3, 0, 1, 2)

        # 1 C T H W.
        inputs = inputs.unsqueeze(0)

        # Sample frames for the fast pathway.
        index = torch.linspace(0, inputs.shape[2] - 1, self.frame_length).long()
        fast_pathway = torch.index_select(inputs, 2, index)
        # logger.info('fast_pathway.shape={}'.format(fast_pathway.shape))

        # Sample frames for the slow pathway.
        index = torch.linspace(0, fast_pathway.shape[2] - 1,
                               fast_pathway.shape[2] // self.slowfast_alpha).long()
        slow_pathway = torch.index_select(fast_pathway, 2, index)
        # logger.info('slow_pathway.shape={}'.format(slow_pathway.shape))
        inputs = [slow_pathway.to(self.device), fast_pathway.to(self.device)]

        return slow_pathway.to(self.device).cpu().detach().numpy(), fast_pathway.to(self.device).cpu().detach().numpy()

    def save_frames(self, frame):
        if len(self.saved_frames) == self.sqn_length/2:
            self.midframe_resized = cv2.resize(frame, (640, 480), cv2.INTER_AREA)
            self.midframe = frame
        frame = cv2.resize(frame, (640, 480), cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = scale(256, frame)
        self.saved_frames.append(frame)

    def check_save_frames(self):
        return True if len(self.saved_frames) == self.sqn_length else False

    def clear_frame_space(self):
        self.saved_frames = []

    def boxes(self, frame, resized=True):
        self.height, self.width, self.channel = frame.shape
        outputs = self.predictor(frame)
        fields = outputs["instances"]._fields
        pred_classes = fields["pred_classes"]
        selection_mask = pred_classes == 0
        # acquire person boxes
        pred_boxes = fields["pred_boxes"].tensor[selection_mask]
        boxes = cv2_transform.scale_boxes(self.crop_size,
                                          pred_boxes,
                                          self.height,
                                          self.width).to(self.device)
        empty = torch.zeros((boxes.shape[0], 1)).to(self.device)
        empty = [empty, boxes]
        boxes = torch.cat(empty, dim=1)
        boxes = boxes.cpu().detach().numpy()
        if not resized:
            ratio = np.min(
                [self.height, self.width]
            ) / 256
            ori_size_boxes = boxes[:, 1:] * ratio
            self.new_boxes = ori_size_boxes
            return None
        return boxes

    def action_recognition(self, frame):
        self.save_frames(frame)
        if self.check_save_frames():
            slow, fast = self.preprocess(self.saved_frames)
            # chose mid point to extract boxes, can use start of clip / end of clip / any point of the clip
            boxes = self.boxes(self.midframe_resized, resized=True)
            print(self.midframe_resized.shape)
            self.boxes(self.midframe, resized=False)
            self.clear_frame_space()
            if boxes.size != 0:
                output = self.session.run([self.output_name], {self.input_name: slow, self.input_name2: fast, self.input_name3: boxes})
                self.preds = np.array(output[0])
                self.result()

    def result(self, confidence=0.7):
        if self.preds.size != 0:
            pred_masks = self.preds > confidence
            label_ids = [np.nonzero(pred_mask)[0] for pred_mask in pred_masks]
            pred_labels = [
                [self.class_names[label_id] +str("  ") +str(self.preds[0][label_id]) for label_id in perbox_label_ids]
                for perbox_label_ids in label_ids
            ]
            self.pred_labels = pred_labels
            return pred_labels

    def draw_box_on_frame(self, frame):
        if self.new_boxes.size != 0 and self.pred_labels is not None:
            for box, box_labels in zip(self.new_boxes.astype(int), self.pred_labels):
                cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), thickness=2)
                label_origin = box[:2]
                for o,label in enumerate(box_labels):
                    label_origin[-1] -= 5
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 2)
                    cv2.rectangle(
                        frame,
                        (label_origin[0], label_origin[1] + 5),
                        (label_origin[0] + label_width, label_origin[1] - label_height - 5),
                        self.palette[o], -1
                    )
                    cv2.putText(
                        frame, label, tuple(label_origin),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
                    )
                    label_origin[-1] -= label_height + 1
        return frame