import argparse
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torch.nn.functional as F

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

# define the adversarial attack
def adversarial_attack(image, epsilon, target_class):
    # TODO
    # for now, just copy and return input image
    adversarial_image = image.clone().detach()
    return adversarial_image

# generate an adversarial example
def generate_adversarial_image(image_path, epsilon, target_class):
    image = load_image(image_path)
    image.requires_grad = True
    
    # forward pass the original image through the model
    original_output = model(image)
    original_probabilities = F.softmax(original_output, dim=1)
    original_pred = original_output.max(1, keepdim=True)[1]

    # generate image by calling adversarial_attack
    adversarial_image = adversarial_attack(image, epsilon, target_class)

    # calculate noise
    noise = adversarial_image - image

    # classify the perturbed image
    adversarial_output = model(adversarial_image)
    adversarial_probabilities = F.softmax(adversarial_output, dim=1)
    adversarial_pred = adversarial_output.max(1, keepdim=True)[1]

    # now display image(s):

    # convert tensor images to PIL images
    original_image_pil = transforms.ToPILImage()(image.squeeze(0))
    # normalize noise image for visualization
    noise_image_pil = transforms.ToPILImage()(0.5 + 0.5 * noise.squeeze(0))
    adversarial_image_pil = transforms.ToPILImage()(adversarial_image.squeeze(0))
    
    # parsing the class names correctly
    original_class_name = classes[original_pred.item()].split(',')[0].replace("'", "").strip()
    adversarial_class_name = classes[adversarial_pred.item()].split(',')[0].replace("'", "").strip()
    
    print(f'original image: {original_class_name}')
    print(f'adversarial image: {adversarial_class_name}')



generate_adversarial_image(args.image_path, epsilon=0.1, target_class=args.target_class)