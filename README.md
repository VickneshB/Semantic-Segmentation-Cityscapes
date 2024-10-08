
# Cityscapes Semantic Segmentation with U-Net

This project implements a semantic segmentation model using the Cityscapes dataset and a U-Net architecture built with PyTorch. The model is trained to classify various urban scene elements such as roads, vehicles, buildings, and pedestrians from high-resolution images.

## Project Overview

The goal of this project is to perform dense semantic segmentation on urban street scenes using the Cityscapes dataset. The model is built using a U-Net architecture and optimized for high Mean Intersection over Union (MeanIoU) across 19 valid classes.

### Key Features

- **Model Architecture:** U-Net with a series of convolutional down-sampling and up-sampling layers.
- **Dataset:** Cityscapes dataset, which includes 5000 finely annotated images from urban environments.
- **Training with CUDA Support:** The project is designed to run on a GPU for faster training using CUDA if available.
- **Custom DataLoader:** A PyTorch `Dataset` class that efficiently loads and preprocesses Cityscapes images and their corresponding labels.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install the required dependencies:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install albumentations opencv-python-headless matplotlib cityscapesscripts tqdm
   ```

3. **Download the Cityscapes dataset:**
   - Follow the instructions at [Cityscapes Dataset](https://www.cityscapes-dataset.com) to download the dataset.
   - Place the dataset in the appropriate directories for images and labels.

## Dataset Structure

The dataset should be organized as follows:

```bash
├── leftImg8bit_trainvaltest/
│   ├── leftImg8bit
│      ├── train/
│      ├── val/
│      ├── test/
├── gtFine_trainvaltest/
│   ├── gtFine
│      ├── train/
│      ├── val/
│      ├── test/
```

## Model Training

1. **Prepare the dataset:**
   Ensure the images and corresponding label masks are placed correctly in the directories.

2. **Start training:**
   The following steps are included in the notebook:
   - Dataset preprocessing and augmentation.
   - Defining the U-Net model architecture.
   - Training loop with performance metrics (e.g., loss, MeanIoU).

   Run the cells in the notebook to start the training process. Make sure you have a compatible GPU for efficient training.

## Usage

- **Training the Model:** Modify the paths in the notebook for your specific dataset location and run the provided training code.
- **Evaluating Performance:** The notebook includes code to evaluate the trained model on validation data using metrics like MeanIoU.
- **Visualization:** You can visualize the predicted segmentation masks alongside ground truth labels using the provided visualization tools in the notebook.

## Results

The model is designed to achieve high accuracy in segmenting various objects in urban scenes, and detailed results will be logged during training. The notebook also supports visualizing predictions.

## Acknowledgments

- Cityscapes dataset: [Cityscapes Dataset](https://www.cityscapes-dataset.com)
- PyTorch U-Net implementation inspired by multiple sources and enhanced for semantic segmentation tasks.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
