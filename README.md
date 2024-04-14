# Adversarial Noise Generator

This is a tool designed to manipulate images such that they look roughly the same to a human, but are misclassified by a pretrained model (resnet50, in this case). You input an image and a specified target class, and it outputs a slightly altered copy of the image.

## Usage Instructions

### Setup

1. **Prepare Images**: 
   - You can either use your own images by placing them in the `sample_images` directory.
   - Alternatively, use the default images provided in the `sample_images` folder.

### Generate Adversarial Image

```bash
python generate.py <image_name> <target_class_number>
```

- **image_name**: the name of the image file you wish to alter.
- **target_class_number**: the numerical identifier for the target class. The class numbers are mapped in the imagenet_classes.txt.

Example:

```bash
python generate.py panda.jpg 368
```

This command will modify the panda.jpg image so that it's classified as a Gibbon (class number 368) by the model.