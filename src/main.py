import torch
import torch.nn.functional as F
from torchvision import transforms
import gradio as gr
from PIL import Image
from model import DigitNet

# Prediction function
def predict_digit(img):
    img = transform(img).unsqueeze(0)  # shape: [1, 1, 28, 28]
    with torch.inference_mode():
        logits = model(img)
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        print(f"Prediction: {predicted_class}")
    return {str(i): float(probs[0][i]) for i in range(10)}

if __name__ == "__main__":
    # Load the model
    model = DigitNet()
    model.load_state_dict(torch.load("models/model_weights.pth", weights_only=True))
    model.eval()

    # Create transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(
        height=280,
        width=280, 
        image_mode="L"
        ),
    outputs=gr.Label(num_top_classes=10),
    live=True
    )

    interface.launch()
