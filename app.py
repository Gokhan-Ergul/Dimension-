import gradio as gr
import torch
import timm
import pandas as pd
import numpy as np
from torchvision import transforms
import matplotlib.patches as patches
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import os
from groundingdino.util.inference import load_model, load_image, predict, annotate
import glob
from huggingface_hub import hf_hub_download
import shutil

import warnings
warnings.filterwarnings('ignore')


class SwinCSVRegressor(nn.Module):
    def __init__(self, num_csv_features):
        super(SwinCSVRegressor, self).__init__()
        
        # Swin Transformer backbone
        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)

        swin_out_dim = self.swin.num_features # will be 768

        # CSV MLP
        self.csv_mlp = nn.Sequential(
            nn.Linear(num_csv_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            #nn.ReLU(), 
        )

        # Combined regressor
        self.regressor = nn.Sequential(
            nn.Linear(swin_out_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # Output: [real_width, real_height]
            #nn.ReLU()  #  This ensures output >= 0
        )

    def forward(self, image, csv_features):
        img_features = self.swin(image)  # shape: [batch_size, 768]
        
        if len(img_features.shape) == 4:
            img_features = img_features.mean(dim=[2, 3])
        csv_features = self.csv_mlp(csv_features)
        combined = torch.cat((img_features, csv_features), dim=1)
        output = self.regressor(combined)
        return output




class ImageCSVRegressionDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = csv_file
        self.image_dir = image_dir
        self.transform = transform
        
        # Drop target columns to use as CSV features
        self.csv_features = self.df.drop(['real_width', 'real_height'], axis=1)
        
        # Extract labels
        self.labels = self.df[['real_width', 'real_height']].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_pil = Image.fromarray(self.image_dir)         # NumPy → PIL
        image = image_pil.convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # CSV features (without image_id, remove targets too)
        csv_row = self.csv_features.iloc[idx]
        
        # Optionally drop columns you don't need like image_id, image_width/height
        csv_row = csv_row.drop(['image_id'])  # Drop more if needed
        
        csv_tensor = torch.tensor(csv_row.values, dtype=torch.float32)
        
        # Labels
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, csv_tensor, label

def resize_with_padding(image_paths=None, target_size=(224, 224), image=None):
    padded_images = []
    if image_paths is not None:
        
        for img_path in image_paths:
            # Load the image
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Image at {img_path} could not be loaded.")
            
            old_size = image.shape[:2]  # (height, width)
            ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    
            new_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))  # (width, height)
            resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    
            mean_color = list(map(int, resized_image.mean(axis=(0,1))))
            delta_w, delta_h = target_size[1]-new_size[0], target_size[0]-new_size[1]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
    
            padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=mean_color)
    
            padded_images.append({'image': padded_image, 'path': img_path})
        
        return padded_images

    if image is not None:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        if image.ndim == 2:  # Eğer siyah-beyaz ise, 3 kanal yap
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        old_size = image.shape[:2]  # (height, width)
        ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
        new_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))  # (width, height)
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        mean_color = list(map(int, resized_image.mean(axis=(0,1))))
        delta_w, delta_h = target_size[1]-new_size[0], target_size[0]-new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=mean_color)
        return padded_image

def test_model(model, test_loader, device, target_scale=180, real_height = None, real_width = None):
    model.eval()

    if real_height and real_width:  # Eğer boş değilse
        try:
            real_width = float(real_width) if real_width not in [None, ''] else 0.0
            real_height = float(real_height) if real_height not in [None, ''] else 0.0
        except ValueError:
            real_width = 0.0
            real_height = 0.0
    else:
        real_width = 0.0
        real_height = 0.0
        
    predictions = ""
    
    with torch.no_grad():
        for images, csv_feats, targets in test_loader:
            targets = torch.tensor([[real_width, real_height]], dtype=torch.float32)
                
                
            images = images.to(device)
            csv_feats = csv_feats.to(device)
            targets = targets.to(device)

            outputs = model(images, csv_feats)
            outputs = torch.abs(outputs)
            outputs = outputs * target_scale

            test_cm_error = torch.abs(outputs - targets).mean().item()

            width_pred, height_pred = outputs[0].tolist()
            
        if real_height != 0.0 and real_width != 0.0:
            predictions += f"""
            # <span style="color: #FF5733;">**Predictions**</span>

            ## <span style="color: #C70039;">**Avg CM Error**</span>: <span style="color: #c70202;">{test_cm_error:.2f} cm</span>

            ### <span style="color: #C70039;">**Real dimensions**</span>:
            - <span style="color: #FF5733;">**Width**</span>: <span style="color: #c70202;">{real_width:.2f} cm</span>
            - <span style="color: #FF5733;">**Height**</span>: <span style="color: #c70202;">{real_height:.2f} cm</span>

            ### <span style="color: #C70039;">**Predicted dimensions**</span>:
            - <span style="color: #FF5733;">**Width**</span>: <span style="color: #c70202;">{width_pred:.2f} cm</span>
            - <span style="color: #FF5733;">**Height**</span>: <span style="color: #c70202;">{height_pred:.2f} cm</span>
            """
        else:
            predictions += f"""
            # <span style="color: #FF5733;">**Predictions**</span>

            ## <span style="color: #C70039;">**Predicted dimensions**</span>:
            - <span style="color: #FF5733;">**Width**</span>: <span style="color: #c70202;">{width_pred:.2f} cm</span>
            - <span style="color: #FF5733;">**Height**</span>: <span style="color: #c70202;">{height_pred:.2f} cm</span>
            """

    return predictions





def process_image(img, product_name, real_height = None, real_width = None):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_model_path = "models/best_model.pth"

        if not os.path.exists(local_model_path):
            downloaded_path = hf_hub_download(
                repo_id="gokhanErgul/my-348mb-model",
                filename="best_model_epoch34_valCMErr7.62.pth"
            )
            os.makedirs("models", exist_ok=True)
            
            
            shutil.copy(downloaded_path, local_model_path)

        num_csv_features = 12
        model = SwinCSVRegressor(num_csv_features=num_csv_features)
        model.load_state_dict(torch.load(local_model_path, map_location=device))


        #model_path_swin = hf_hub_download(repo_id="gokhanErgul/my-348mb-model", filename="best_model_epoch34_valCMErr7.62.pth")
        
        #model = torch.load(model_path, map_location=device)

        rreall_height, rreall_width = cv2.imread(img).shape[:2]
        model_path = "groundingdino_swint_ogc.pth"
        config_path = "GroundingDINO_SwinT_OGC.py"
        #model_path_swin = 'best_model_epoch34_valCMErr7.62.pth'
        last_csv = 'df_cropped_csv_last_version.csv'
        # Load swin Model for measureing.
        
        #model.load_state_dict(torch.load(model_path_swin, map_location=device))
        model.to(device)
        model.eval()
    
        # Load the Model for object detection.
        model_groundingdino = load_model(config_path, model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_groundingdino.to(device)
        
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        df_cropped_csv_last_version = pd.read_csv(last_csv)
        scaler = StandardScaler()
        
        csv_features_scaled = scaler.fit_transform(
            df_cropped_csv_last_version.drop(['real_width', 'real_height', 'image_id'], axis=1)
        )
        
        df_box = pd.DataFrame()
        df_box['image_id'] = 0
        df_box['image_width'] = 0
        df_box['image_height'] = 0
        df_box['bounding_box_width'] = 0.0
        df_box['bounding_box_height'] = 0.0
        df_box['box_relative_width'] = 0.0  # bounding_box_width / image_width
        df_box['box_relative_height'] = 0.0  # bounding_box_height / image_height
        df_box['confidence'] = 0.0  # Confidence score
        df_box['box_area'] = 0.0
        df_box['bbox_aspect_ratio'] = 0  # bounding_box_width / bounding_box_height
        df_box['bbox_diag'] = 0  # sqrt(bw²+bh²)
        df_box['log_box_area'] = 0  # log_box_area
        df_box['norm_confidence'] = 0  # Normalized confidence
        df_box['real_height'] = 0
        df_box['real_width'] = 0
        df_box.head()
        
        img_n = cv2.imread(img)
        img_rgb = cv2.cvtColor(img_n, cv2.COLOR_BGR2RGB)
        img_reszied = resize_with_padding(image=img_rgb)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(img_reszied)
        image = img_reszied
        
        device = next(model_groundingdino.parameters()).device
        image_tensor = image_tensor.to(device=device, dtype=torch.float32)
        
        #model pridect 
        boxes, logits, phrases = predict(
            model=model_groundingdino,
            image=image_tensor,
            caption= product_name,
            box_threshold=0.3,
            text_threshold=0.25,
            device=str(device)
        )
        
        fig, ax = plt.subplots(1, figsize=(8, 6))
        ax.imshow(image)
        
        if logits is None or len(logits) == 0:
            raise ValueError(f"No object detected in image: {img}. Please check the image you sent or the object name you entered.") 
            
        max_logit_index = int(logits.argmax())
        
        for i, box in enumerate(boxes):
        
            x_center, y_center, width, height = box.tolist()  
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
        
            image_height, image_width = image.shape[:2]
            x_min = int(x_min * image_width)
            y_min = int(y_min * image_height)
            x_max = int(x_max * image_width)
            y_max = int(y_max * image_height)
        
            width = x_max - x_min
            height = y_max - y_min
            
            if i == max_logit_index:
                ax.add_patch(patches.Rectangle((x_min, y_min), width, height, linewidth=3, edgecolor='green', facecolor='none'))
            else:
                ax.add_patch(patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none'))
            
            y_text = y_min - height * 0.05
        
            if i == max_logit_index:
                ax.text(x_min , y_text , f'{phrases[i]}: {logits[i]:.2f}', color='green', fontsize=10, verticalalignment='center')
            else:
                ax.text(x_min , y_text , f'{phrases[i]}: {logits[i]:.2f}', color='red', fontsize=10, verticalalignment='center')
            
        
        
        saved_fig_boxes = fig
        saved_ax_boxes = ax
        saved_ax_boxes.set_title('Found Objects')
        saved_ax_boxes.axis('off')
                   
        best_index = int(logits.argmax())
        best_box = boxes[best_index].unsqueeze(0)
        
        #detected the area
        x_center, y_center, width, height = best_box[0].tolist()
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        
        image_height, image_width = image.shape[:2]
        
        #crop
        x_min = int(x_min * image_width)
        y_min = int(y_min * image_height)
        x_max = int(x_max * image_width)
        y_max = int(y_max * image_height)
        
        main_x_min, main_y_min, main_x_max, main_y_max = x_min, y_min, x_max, y_max
        
        json_output = {
        "X min": main_x_min,
        "Y min": main_y_min,
        "X max": main_x_max,
        "Y max": main_y_max,
        }
        import json
        json_string = json.dumps(json_output)
              
        crop = image[y_min:y_max, x_min:x_max]
        
        main_crop_height = y_max - y_min
        main_crop_width = x_max - x_min
        
        product_name = product_name.strip().replace(' ', '_')
        
        h,w = crop.shape[:2]
        box_area = w * h
        bbox_aspect_ratio = w / h if h != 0 else 0
        bbox_diag = np.sqrt(w**2 + h**2)
        
        log_box_area = np.log(box_area) if box_area > 0 else 0
        logits = logits
        confidence = logits.max().item()
        norm_confidence = confidence / logits.sum().item() if logits.sum().item() > 0 else 0
                       
        # detection found
        if logits.shape[0] > 0:
            df_box.loc[len(df_box)] = [
                101,
                140,
                image_height,
                w,
                h,
                w / image_width,
                h / image_height,
                confidence,
                box_area,
                bbox_aspect_ratio,
                bbox_diag,
                log_box_area,
                norm_confidence,
                0,
                0
            ]
        else:  # no detection
            df_box.loc[len(df_box)] = [
                image_width,
                image_height,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        
        padded = resize_with_padding(image=crop, target_size=(224, 224))
        
        new_path = f"{product_name}_{'101'}.jpg"
        fig_crop, ax_crop = plt.subplots(figsize= (6,6))
        ax_crop.imshow(padded)
        
        ax_crop.axis('off')
        ax_crop.set_title("Cropped Image")
        saved_ax_cropped = ax_crop
        saved_fig_cropped = fig_crop
    
        df_box_scaled = scaler.transform(
        df_box.drop(columns=['image_id','real_width','real_height'],axis=1)
        )
        df_box_new = df_box.copy()     
        df_box_new.loc[:, df_box_new.columns.difference(['real_width', 'real_height', 'image_id'])] = df_box_scaled
        
        df_box['image_width'] = rreall_width
        df_box['image_height'] = rreall_height
        df_box['real_height'] = real_height
        df_box['real_width'] = real_width
    
        csv_file = df_box_new.iloc[[-1]]
        image_dir = padded
        
        dataset = ImageCSVRegressionDataset(
            csv_file=csv_file,
            image_dir=image_dir,
            transform=transform #this only applies to images
        )
        
        test_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        predictions = test_model(model, test_loader, device,real_height = real_height, real_width = real_width)
    
        return saved_fig_boxes, saved_fig_cropped, json_output , df_box, predictions

    except Exception as e:
        gr.Error(str(e))
        return None, None, None, None, gr.update(value=f"❌ Error: {str(e)}", visible=True)


css = """

body {
    font-family: 'Segoe UI', sans-serif;
    background-color: #4b5661;
    color: #1f2937;
}


.gr-box, .gr-column, .gr-group {
    background-color: white !important;
    border-radius: 18px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    padding: 16px;
    transition: box-shadow 0.3s ease;
}

.gr-box:hover {
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.08);
}

.custom-plot canvas {
    height: 320px !important;
    width: 100% !important;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

input[type="text"], textarea, .gr-textbox, .gr-json {
    border: 1px solid #d1d5db;
    border-radius: 10px;
    padding: 10px;
    font-size: 15px;
    background-color: #92a2b3;
    transition: border-color 0.2s ease;
}

input[type="text"]:focus, textarea:focus {
    border-color: #6366f1;
    outline: none;
}

.gr-dataframe-component {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e5e7eb;
}

.submit-btn, .clear-btn {
    padding: 14px 28px;
    border: none;
    font-size: 16px;
    font-weight: 600;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.25s ease-in-out;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

.submit-btn {
    background: linear-gradient(to right, #6366f1, #4f46e5);
    color: white !important;
}

.submit-btn:hover {
    background: linear-gradient(to right, #4f46e5, #4338ca);
    transform: scale(1.03);
}

.clear-btn {
    background: linear-gradient(to right, #ef4444, #dc2626);
    color: white !important;
}

.clear-btn:hover {
    background: linear-gradient(to right, #dc2626, #b91c1c);
    transform: scale(1.03);
}

.gr-markdown-content {
    background-color: #f1f5f9;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    font-size: 15px;
}
"""



with gr.Blocks(css = css , theme=gr.themes.Default()) as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="📷 Input Image", height=360, width=640)
            product_name = gr.Textbox(label="🔤 Product Name", placeholder="e.g. Black glasses case")
            real_height = gr.Textbox(label="📏 Real Height (cm)", placeholder="Optional:")
            real_width = gr.Textbox(label="📐 Real Width (cm)", placeholder="Optional:")
            with gr.Row():
                clear_btn = gr.Button("🧹 Clear", size="lg", elem_classes="clear-btn")
                submit_btn = gr.Button("🚀 Submit", size="lg", elem_classes="submit-btn")
                
        
        with gr.Column(scale=1.2): 
            detection_plot = gr.Plot(label="🧭 Object Detection", elem_classes="custom-plot")
            box_coords = gr.JSON(label="📦 Bounding Box Coordinates")
            cropped_image = gr.Plot(label="✂️ Cropped Image",elem_classes="custom-plot")
            features = gr.DataFrame(label="📊 Extracted Features")
            prediction_text = gr.Markdown()

    submit_btn.click(
        fn=process_image,
        inputs=[image_input, product_name, real_height, real_width],
        outputs=[detection_plot, cropped_image, box_coords, features, prediction_text]
    )
    clear_btn.click(
        fn=lambda: (None, "", "", "", None, None, None, None, ""),
        inputs=[],
        outputs=[
            image_input, product_name, real_height, real_width,
            detection_plot, cropped_image, box_coords, features, prediction_text
        ]
    )

demo.launch()

