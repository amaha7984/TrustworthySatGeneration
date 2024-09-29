import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import os
import base64
from sendgrid_notification import send_email_notification


# Streamlit Class Definition
class StreamlitApp:
    def __init__(self):
        # Define model path
        self.model_path = '/Users/arpanmahara/CNNDetection/checkpoints/sat2map_kanattn/model_epoch_best.pth'
        # Set up preprocessing parameters
        self.no_resize = True
        self.no_crop = False
        self.load_size = (256, 256)  # Resize dimensions if enabled
        self.crop_size = 224  # Crop size if enabled
        # Load the model when initializing
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model()
        

    def load_model(self):
        """ Load the trained ResNet50 model for binary classification """
        model = models.resnet50(num_classes=1)
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict['model'])
        model = model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, image):
        """ Preprocess the uploaded image for model inference """
        transformations = []

        # Resize if not disabled
        if not self.no_resize:
            transformations.append(transforms.Resize(self.load_size))

        # Crop if not disabled
        if not self.no_crop:
            transformations.append(transforms.CenterCrop(self.crop_size))

        # Standard to-tensor and normalization
        transformations.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Compose all transformations
        transform_pipeline = transforms.Compose(transformations)

        # Apply transformations to the image
        image_tensor = transform_pipeline(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(self.device)

    def predict_image(self, image_tensor):
        """ Run inference on a single image tensor """
        with torch.no_grad():
            output = self.model(image_tensor)
            predicted = torch.sigmoid(output).item()  # Use sigmoid for binary classification
            return 'Generated' if predicted >= 0.5 else 'Real'
        
    def get_base64_image(self, file_path):
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    def run(self):
        image_path = "/Users/arpanmahara/CNNDetection/data/neuralnetworks.jpg"
        image_base64 = self.get_base64_image(image_path)
        custom_css = f"""
        <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{image_base64}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .scrolling-title {{
                font-size: 32px;
                font-weight: bold;
                color: yellow; 
                overflow: hidden;
                white-space: nowrap;
                box-sizing: border-box;
                animation: scroll-left 10s linear infinite;
                text-align: center;
                background: rgba(0, 0, 0, 0.5); /* Transparent background for title */
                padding: 10px;
                border-radius: 8px;
            }}
            @keyframes scroll-left {{
                0% {{
                    transform: translateX(100%);
                }}
                100% {{
                    transform: translateX(-100%);
                }}
            }}
            
            .upload-label {{
                color: white;
                font-size: 18px;
            }}
            .prediction-text {{
                color: white;
                font-weight: bold;
                font-size: 24px;
            }}
            .center img {{
                display: block;
                margin: 0 auto;
                width: 400px;
                }}
                
            .center-text {{
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 24px;
                color: white;
                font-weight: bold;
                margin-top: -10px;
            }}
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown('<div class="scrolling-title">Forensic Detection of Satellite Image Forgery</div>', unsafe_allow_html=True)

        # st.write("Please Upload an image to check if it is real or generated.")

        # File uploader in Streamlit
        st.markdown('<p class="upload-label">Please Upload an image to check if it is real or generated.</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="uploader")

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            base64_image = base64.b64encode(file_bytes).decode()

            # Center-align the image using custom HTML and CSS
            st.markdown(f'<div class="center"><img src="data:image/png;base64,{base64_image}" alt="Uploaded Image"></div>', unsafe_allow_html=True)

            # Reset file pointer to the beginning after base64 read
            uploaded_file.seek(0)

            # Reload the image for prediction
            image = Image.open(uploaded_file).convert('RGB')

            st.write("")
            image_tensor = self.preprocess_image(image)

            # Predict using the model
            result = self.predict_image(image_tensor)

            # Display the prediction
            st.markdown(f'<div class="center-text">The image is predicted as: **{result}**</div>', unsafe_allow_html=True)
            
            if result == 'Generated':
                send_email_notification(result, recipient_email="maharaarpan1212@gmail.com")
                st.success(f"Email alert sent to maharaarpan1212@gmail.com as the image is detected as: {result}!")

# Create and run the Streamlit app
if __name__ == "__main__":
    # Initialize Streamlit App
    app = StreamlitApp()
    # Run the app
    app.run()
