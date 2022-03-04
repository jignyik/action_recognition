import numpy as np
import cv2
import torch
import onnxruntime
import math


class PreProcessing:
    def __init__(self):
        self.device = "cpu"
        self.saved_frames = []
        self.frame_length = 32
        self.sample_rate = 2
        self.slowfast_alpha = 4
        self.sqn_length = self.frame_length*self.sample_rate

        self.action_model, self.boxes_model, self.action_input_names, self.action_output_names, self.box_input_names\
            , self.box_output_names = self.load_onnx_session()
        self.class_names = self.load_action_class_names()
        self.height, self.width, self.channel = None, None, None
        self.new_boxes = np.array([])
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
        self.crop_size = 256
        self.pred_labels = None
        self.palette = np.random.randint(64, 128, (len(self.class_names), 3)).tolist()
        self.preds = np.array([])

    @staticmethod
    def load_onnx_session():
        action = onnxruntime.InferenceSession(r"model.onnx")
        boxes = onnxruntime.InferenceSession(r"FasterRCNN-10.onnx")
        action_input_name = [i.name for i in action.get_inputs()]
        action_output_name = [i.name for i in action.get_outputs()]
        boxes_input_name = [i.name for i in boxes.get_inputs()]
        boxes_output_name = [i.name for i in boxes.get_outputs()]
        return action, boxes, action_input_name, action_output_name, boxes_input_name, boxes_output_name

    @staticmethod
    def load_action_class_names():
        path_to_csv = "ava.names"
        with open(path_to_csv) as f:
            labels = f.read().split('\n')[:-1]
        return labels

    @staticmethod
    def scale(size, image):
        """
        Scale the short side of the image to size.
        Args:
            size (int): size to scale the image.
            image (array): image to perform short side scale. Dimension is
                `height` x `width` x `channel`.
        Returns:
            (ndarray): the scaled image with dimension of
                `height` x `width` x `channel`.
        """
        height = image.shape[0]
        width = image.shape[1]
        if (width <= height and width == size) or (
                height <= width and height == size
        ):
            return image
        new_width = size
        new_height = size
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
        else:
            new_width = int(math.floor((float(width) / height) * size))
        img = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )
        return img.astype(np.float32)

    @staticmethod
    def scale_boxes(size, boxes, height, width):
        """
        Scale the short side of the box to size.
        Args:
            size (int): size to scale the image.
            boxes (ndarray): bounding boxes to peform scale. The dimension is
            `num boxes` x 4.
            height (int): the height of the image.
            width (int): the width of the image.
        Returns:
            boxes (ndarray): scaled bounding boxes.
        """
        if (width <= height and width == size) or (
                height <= width and height == size
        ):
            return boxes

        new_width = size
        new_height = size
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
            boxes *= float(new_height) / height
        else:
            new_width = int(math.floor((float(width) / height) * size))
            boxes *= float(new_width) / width
        return boxes

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
        if len(self.saved_frames) == int(self.sqn_length/2):
            self.new_boxes = self.boxes(frame)
            if len(self.new_boxes) == 0:
                self.clear_frame_space()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.scale(256, frame)
        self.saved_frames.append(frame)

    def check_save_frames(self):
        return True if len(self.saved_frames) == self.sqn_length else False

    def clear_frame_space(self):
        self.saved_frames = []

    @staticmethod
    def boxes_preprocess(image):
        # Resize
        ratio = 800.0 / min(image.shape[0], image.shape[1])
        image = cv2.resize(image, (int(ratio * image.shape[1]), int(ratio * image.shape[0])), cv2.INTER_LINEAR).astype(
            "float32")

        # HWC -> CHW
        image = np.transpose(image, [2, 0, 1])
        # Normalize
        mean_vec = np.array([102.9801, 115.9465, 122.7717])
        for i in range(image.shape[0]):
            image[i, :, :] = image[i, :, :] - mean_vec[i]

        # Pad to be divisible of 32
        import math
        padded_h = int(math.ceil(image.shape[1] / 32) * 32)
        padded_w = int(math.ceil(image.shape[2] / 32) * 32)

        padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
        padded_image[:, :image.shape[1], :image.shape[2]] = image
        image = padded_image

        return image

    def boxes(self, frame, threshold=0.8):
        img_data = self.boxes_preprocess(frame)
        output = self.boxes_model.run(output_names=self.box_output_names, input_feed={self.box_input_names[0]: img_data})
        ratio = 800.0 / min(frame.shape[0], frame.shape[1])
        mask = (output[2] > threshold) & (output[1] == 1)
        out = (output[0][mask]/ratio)
        n = len(out)
        if n == 0:
            return out
        out = np.concatenate((np.zeros((n, 1)), out), axis=1).astype("float32")
        self.new_boxes = out
        return out

    def action_recognition(self, frame):
        self.save_frames(frame)
        if self.check_save_frames():
            slow, fast = self.preprocess(self.saved_frames)
            # chose mid point to extract boxes, can use start of clip / end of clip / any point of the clip
            self.clear_frame_space()
            output = self.action_model.run(self.action_output_names,
                                           {self.action_input_names[0]: slow,
                                            self.action_input_names[1]: fast,
                                            self.action_input_names[2]: self.new_boxes})
            self.preds = np.array(output[0])
            self.result()

    def result(self, confidence=0.5):
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
                cv2.rectangle(frame, tuple(box[1:3]), tuple(box[3:]), (0, 255, 0), thickness=2)
                label_origin = box[1:3]
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