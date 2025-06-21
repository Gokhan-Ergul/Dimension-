# 📏 Visual Estimation of Real-World Product Dimensions

> Estimate the physical height & width of a product — just from an user taken image.

---

## 🎯 Research Objective

Can we predict a product’s real-world dimensions **without any measurement input** — using only user-taken photos?

This project explores the limits of what kind of **physical insight** can be extracted from **visual-only signals**, especially in noisy, real-world e-commerce scenarios.

---

## 🧹 Data Collection & Cleaning

- Scraped **24,000+ user-uploaded images** from an e-commerce platform.
- Covered **520 product categories**, ~60 images per product.
- Images were **noisy and inconsistent**, as expected from user content.

✅ A custom visual filtering pipeline was implemented to **automatically retain only clean product views**, replacing the need for manual curation.

---

## 🔍 Object Detection: Finding the Product

Bounding box annotations were not available, so I experimented with several **zero-shot / low-shot object detection** approaches:

- VGG16-based Visual Outlier Detection  
- YOLOv8  
- CLIP + SAM  
- ✅ **GroundingDINO** (selected: best performance)

> This allowed the pipeline to isolate only the product in each photo, which significantly improved downstream predictions.

---

## 📐 Feature Extraction

Beyond raw image input, each segmented product was used to compute **12 visual-statistical features**, including:

- Aspect ratio  
- Normalized area  
- Rectangularity  
- Center offset  
- Foreground-background contrast  
- and more

> These structured features act as a helpful inductive bias alongside image-based learning.

---

## 🧠 Modeling: Dimension Regression with Deep Learning

**Input:** Cropped product image + 12D feature vector  
**Output:** Real-world height & width (float regression)

### 🏗️ Models Evaluated:

- ResNet50  
- EfficientNetB3  
- ConvNeXt  
- ✅ **Swin Transformer** (best performer)

After identifying Swin as the top model, I applied a **two-phase training** strategy:

1. **Frozen backbone:** Only the regression head was trained initially.  
2. **Unfrozen fine-tuning:** Full model was then fine-tuned end-to-end.

> This improved stability and reduced overfitting in early training.

---

## 🔧 Why PyTorch?

- Native support for research-centric models: CLIP, SAM, GroundingDINO  
- Dynamic computation graphs  
- Easier experimentation and debugging  
- Rapid prototyping with Hugging Face, timm, and segmentation libraries

---

## 🚀 Outcomes & Use Cases

This pipeline combines **object detection, visual feature engineering, and deep regression** to estimate real-world product sizes **from visual signals only**.

### Potential Applications:

- 🛍️ **E-commerce auto-tagging** (dimensions, volume, proportions)  
- 📦 **Packaging optimization** (logistics, shipping cost estimation)  
- 📱 **Mobile apps** (dimension from photo, DIY tools)  
- 🧾 **Metadata generation** for large-scale product databases

---

## 🔗 Try It Yourself

🧪 Live Demo (Hugging Face Space):  
**[👉 Launch Demo](https://huggingface.co/spaces/your-demo-link)**

📘 Full Notebook (Kaggle):  
**[📎 View on Kaggle](https://kaggle.com/your-notebook-link)**

💻 Source Code (GitHub):  
**[💾 GitHub Repository](https://github.com/your-repo-link)**

---

## 📣 Contact

Open to collaboration and feedback!  
Feel free to reach out via GitHub or connect on [LinkedIn](https://www.linkedin.com/in/g%C3%B6khan-erg%C3%BCl-20a827273/).

