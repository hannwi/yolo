import os.path as osp


model_root = "./checkpoints"
models = {
    "face_yolov8n.pt": osp.join(model_root, "face_yolov8n.pt"),
    "face_yolov8s.pt": osp.join(model_root, "face_yolov8s.pt"),
    "hand_yolov8n.pt": osp.join(model_root, "hand_yolov8n.pt"),
    "person_yolov8n-seg.pt": osp.join(model_root, "person_yolov8n-seg.pt"),
    "person_yolov8s-seg.pt": osp.join(model_root, "person_yolov8s-seg.pt"),
}
