from super_gradients.training import models

yolo_nas_pose = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
prediction = yolo_nas_pose.predict("images/sample1.png")
prediction.save("images/processed_sample1.png")