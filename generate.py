import argparse
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# initialize the parser
parser = argparse.ArgumentParser(description='Adversarial attack to misclassify an image')
parser.add_argument('image_path', type=str, help='Path to the image file')
parser.add_argument('target_class', type=int, help='Target class for misclassification')

# parse arguments
args = parser.parse_args()

# load ImageNet class labels
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# load resnet50 model 
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# load and preprocess an image
def load_image(image_path):
    # define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    # add batch dimension before returning
    return transform(image).unsqueeze(0)

# iterative FSGM adversarial attack 
def iterative_adversarial_attack(image, epsilon, target_class, num_steps=10):
    adversarial_image = image.clone().detach()
    adversarial_image.requires_grad = True

    for _ in range(num_steps):
        # forward pass the image through the model
        output = model(adversarial_image)
        # calculate loss
        loss = -torch.nn.functional.cross_entropy(output, torch.tensor([target_class], device=output.device))
        # zero all gradients
        model.zero_grad()
        # backward pass to calc gradients
        loss.backward()
        # get datagrad
        data_grad = adversarial_image.grad.data

        # update adversarial image
        sign_data_grad = data_grad.sign()
        adversarial_image = adversarial_image + epsilon * sign_data_grad / num_steps
        adversarial_image = torch.clamp(adversarial_image, 0, 1)

        # detach updated image and require gradients again
        adversarial_image = adversarial_image.detach()
        adversarial_image.requires_grad = True

    return adversarial_image

# generate, display, and save an adversarial image
def generate_adversarial_image(image_path, epsilon, target_class):
    image = load_image(image_path)
    image.requires_grad = True
    
    # forward pass the original image through the model
    original_output = model(image)
    original_probabilities = F.softmax(original_output, dim=1)
    original_pred = original_output.max(1, keepdim=True)[1]
    # get confidence %
    original_conf = original_probabilities.max().item() * 100

    # generate image by calling iterative FGSM attack
    adversarial_image = iterative_adversarial_attack(image, epsilon, target_class)

    # calculate noise
    noise = adversarial_image - image

    # classify the adversarial image
    adversarial_output = model(adversarial_image)
    adversarial_probabilities = F.softmax(adversarial_output, dim=1)
    adversarial_pred = adversarial_output.max(1, keepdim=True)[1]
    adversarial_conf = adversarial_probabilities.max().item() * 100

    '''
    Now let's save the output image and plot the original image, 
    the noise, and the adversarial image for side by side comparison.
    '''

    # convert tensor images to PIL images
    original_image_pil = transforms.ToPILImage()(image.squeeze(0))
    noise_image_pil = transforms.ToPILImage()(noise.squeeze(0))
    adversarial_image_pil = transforms.ToPILImage()(adversarial_image.squeeze(0))

    # save the adversarial image as final output
    adversarial_image_pil.save("adversarial_output_image.png")
    
    # parse the class names
    original_class_name = classes[original_pred.item()].split(',')[0].replace("'", "").strip()
    adversarial_class_name = classes[adversarial_pred.item()].split(',')[0].replace("'", "").strip()

    # plot images and model predictions side by side for comparison
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(original_image_pil)
    axs[0].title.set_text(f"Original: {original_class_name}\n({original_conf:.2f}% confidence)")
    axs[0].axis('off')
    
    axs[1].imshow(noise_image_pil, cmap='gray')
    axs[1].title.set_text("Adversarial Noise")
    axs[1].axis('off')

    axs[2].imshow(adversarial_image_pil)
    axs[2].title.set_text(f"Adversarial: {adversarial_class_name}\n({adversarial_conf:.2f}% confidence)")
    axs[2].axis('off')

    plt.show()

generate_adversarial_image(args.image_path, epsilon=0.02, target_class=args.target_class)