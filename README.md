<div align="left">
  <img src="./logo_wo_bg.png" alt="VisionStudio Logo" width="200">
</div>


# 📦 VisionStudio
**VisionStudio is CLI-based platform that provides integrated management of Vision AI model training, evaluation, visualization, deployment(export), and experiment tracking(MLflow).**  
VisionStudio was designed with various framework extension in mind, starting with Ultralytics(YOLO).


# 🧠 Overview
VisionStudio = Training + Evaluation + Visualization + Export + Tracking


# 🏗️ Architecture
architecture image ...


# 🚀 Features  
• CLI-based integrated workflow  
• MLflow-based experiment management  
• ONNX export (fixed batch / multi-batch support)  
• Framework-independent architecture  
• Release Note and Model Management Features  


# 📁 Project Structure  
project hierarchy image ...  


# ⚙️ Installation  
Modify the environment name and prefix in the environment.yml file  

``` diff
conda env create --file environment.yml  
conda activate "your VS environment"

pip install -r requirements.txt  
```

# 🧠 CLI Usage  
``` diff
> python main.py --help  
```
# 📌 Commands  

| Command      | Description                       |
| ------------ | --------------------------------- |
| train        | Train model                       |
| evaluate     | Evaluate model                    |
| visualize    | Visualize result                  |
| export       | Export ONNX                       |
| log_eval     | Log evaluation result to MLflow   |
| log_release  | Log model release notes to MLflow |
| upload_model | Upload model to MLflow            |


# 🔥 1. Train  
``` diff
</> Bash  
python main.py train train.yaml  
```

**config.yaml**  
``` diff
</> YAML  
framework: ultralytics  

model: base_models/yolo/yolo11n.pt  
dataset: test/data.yaml  

epochs: 50  
imgsz: 640  
batch: 4  

project_dir: outputs/project_vision_01   
proejct_name: exp01   
```

**Output**  
``` diff
outputs/train/exp01/  
 ├ weights/  
 │   ├ best.pt  
 │   └ last.pt  
 ├ results.png  
 ├ args.yaml  
 └ other files ...  
```

# 🔥 2. Evaluate  
``` diff
</> Bash  
python main.py evaluate eval.yaml  

</> YAML  
image_dir: test/images/val  
label_dir: test/labels/val  
class_file: test/class_names.txt  

framework: ultralytics  
  
model_path: outputs/project_vision_01/exp01/weights/best.pt  
nc: 1  
task: detection  

img_sz: 640  
conf_threshold: 0.001  
nms_threshold: 0.6  
dst_dir: outputs/project_vision_01/exp01/weights/best.pt  
result_name: evaluation_result
```

**Output**  
```diff
evaluation_result.txt  
evaluation_result.json
coco format GT, PREDICT files (.json)  
in evaluation work directory  
```

# 🔥 3. Visualize  
</> Bash  
python main.py evaluate eval.yaml  

</> YAML  
framework: ultralytics  
model_path: outputs/project_vision_01/exp01/weights/best.pt  
nc: 1  
task: detection  
img_sz: 640  
conf_threshold: 0.5  
nms_threshold: 0.3  

**Output**  
Infrence result Display  

Example image ... 
