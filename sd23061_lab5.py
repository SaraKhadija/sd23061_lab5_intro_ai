import streamlit as st
import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd
import torch.nn.functional as F

# -------------------------------
# Step 1: Configure Streamlit page
# -------------------------------
st.set_page_config(
    page_title="Image Classification with ResNet18",
    layout="centered"
)

st.title("CPU-Based Image Classification Web App")
st.write("This application uses a pre-trained ResNet18 model to classify images.")

# -------------------------------------
# Step 2 & 3: Import libraries & CPU use
# -------------------------------------
device = torch.device("cpu")
st.write(f"Running on device: **{device}**")

# --------------------------------------------------
# Step 4: Load pre-trained ResNet18 and eval mode
# --------------------------------------------------
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()
model.to(device)

# --------------------------------------------------
# Step 5: Image preprocessing transformations
# --------------------------------------------------
preprocess = weights.transforms()

# --------------------------------------------------
# Step 6: Image upload interface
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # --------------------------------------------------
    # Step 7: Convert image to tensor & inference
    # --------------------------------------------------
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    # --------------------------------------------------
    # Step 8: Softmax & Top-5 predictions
    # --------------------------------------------------
    probabilities = F.softmax(output, dim=1)[0]
    top5_prob, top5_idx = torch.topk(probabilities, 5)

    labels = weights.meta["categories"]
    results = {
        "Class": [labels[idx] for idx in top5_idx],
        "Probability": [float(prob) for prob in top5_prob]
    }

    df = pd.DataFrame(results)

    st.subheader("Top-5 Predictions")
    st.table(df)

    # --------------------------------------------------
    # Step 9: Visualization (Bar Chart)
    # --------------------------------------------------
    st.subheader("Prediction Probability Distribution")
    st.bar_chart(df.set_index("Class"))

# --------------------------------------------------
# Step 10: End of application
# --------------------------------------------------
