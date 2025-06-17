Unsupervised Stereo Vision Depth Estimation in Endoscopy
This project focuses on estimating depth from stereo endoscopy images using an unsupervised learning approach. The goal is to adapt a state-of-the-art stereo matching model (IGEV++) to the medical domain of endoscopic imaging, without relying on ground-truth depth labels.

🔍 Overview
Depth estimation in endoscopy is a crucial step toward reconstructing 3D structures of internal organs, aiding in diagnostics and surgical planning. However, the lack of labeled data poses a challenge, which we address using unsupervised learning techniques.

📌 Methodology
Model: We used the IGEV++ stereo matching model for disparity prediction.

Unsupervised Loss Functions:

Photometric Loss: The predicted left-image disparity was used to warp the right image into the left viewpoint. The loss was computed between the warped and original left images.

Smoothness Loss: Added to encourage smooth and realistic disparity maps, especially in texture-less regions.

Domain Adaptation: Implemented Low-Rank Adaptation (LoRA) to fine-tune the IGEV++ model efficiently on endoscopic image data.

🧠 Key Features
📷 Unsupervised training — no ground-truth disparities required.

🏥 Medical domain adaptation with minimal computational overhead.

🔧 LoRA fine-tuning — lightweight and modular parameter updates.

