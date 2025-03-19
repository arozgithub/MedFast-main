from ultralytics import YOLO
model=YOLO(r"C:\MedFast-main\medfastai3\client\backend\best.pt")
res=model(r"C:\MedFast-main\medfastai3\client\backend\Tr-me_0262.jpg",save=True,show=True,conf=0.2)