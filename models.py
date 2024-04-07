import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import cv2
import tensorrt as trt
from utils import LetterBox, scale_boxes

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})


        self.input_shape = engine.get_binding_shape(0)[2:]
        self.letterbox = LetterBox(new_shape=self.input_shape)

    def infer(self,x:np.ndarray):
        
        self.inputs[0]['host'] = np.ravel(x)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        return data

    def preprocess(self, image):
        image_data = self.letterbox(image = image)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        image_data = np.array(image_data) / 255.0

        return image_data
    
    def postprocess(self, outputs, org_shape, confidence_thres=0.25, iou_thres=0.45):
        outputs = np.reshape(outputs, (84, 8400))
        outputs = np.transpose(outputs)
        rows = outputs.shape[0]

        boxes = []
        scores = []
        class_ids = []
        
        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                box = [
                    outputs[i][0] - (0.5 * outputs[i][2]),
                    outputs[i][1] - (0.5 * outputs[i][3]),
                    outputs[i][2],
                    outputs[i][3],
                ]

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append(box)

        if len(boxes) == 0:
            return [], [], []
        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)
        boxes = scale_boxes(self.input_shape, boxes, org_shape, xywh = True)
        boxes = boxes.astype(int)
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)

        final_boxes = boxes[indices]
        final_scores = scores[indices]
        final_class_ids = class_ids[indices]

        return final_boxes, final_scores, final_class_ids

    def __call__(self, image):
        input_data = self.preprocess(image)

        outputs = self.infer(input_data)[0]

        boxes, scores, classes = self.postprocess(outputs, image.shape)

        return boxes, scores, classes