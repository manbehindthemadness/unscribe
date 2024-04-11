# Unscribe

Unscribe is a Python library for text removal and scrambling in images using LaMa inpainting and CRAFT text detection.

## Diagrams:
### The following techniques were used in the creation of this project:

#### Large Mask Inpainting with Fourier Convolutions:

![Large Mask Inpainting with Fourier Convolutions](https://raw.githubusercontent.com/geekyutao/Inpaint-Anything/main/example/MainFramework.png)
#### Character Region Awareness For Text Detection:

![Large Mask Inpainting with Fourier Convolutions](https://github.com/clovaai/CRAFT-pytorch/raw/master/figures/craft_example.gif)
#### Contour detection:

![opencv](https://docs.opencv.org/4.x/Find_Contours_Original_Image.jpg)![opencv](https://docs.opencv.org/4.x/Find_Contours_Result.jpg)

## Installation

You can install Describe using pip:

```bash
pip install unscribe
```

## Usage

You can replace `"path/to/your/image.jpg"` with the actual path to your image file. This README provides usage examples for both scrambling and removing text from images.

### Scrambling Text

To scramble text in an image, you can use the following code:

```python
from unscribe import Remover
import cv2

# Initialize the Remover with debug and visualization enabled
remover = Remover(
    show_mats=True,
    debug=True,
    lama_refine=True
)

# Define parameters for scrambling text
image_path = "path/to/your/image.jpg"
low_clamp = 0.1
high_clamp = 0.9
mode = "scramble"  # Set mode to "scramble" for text scrambling
passes = 3

# Load the image
image = cv2.imread(image_path)

# Use the load_mat method to scramble text in the image
scrambled_image = remover.load_mat(
    mat=image,
    low_clamp=low_clamp,
    high_clamp=high_clamp,
    mode=mode,
    passes=passes
)

# Display or save the resulting scrambled image
cv2.imshow("Text Scrambled", scrambled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
<img width="832" alt="lan" src="https://github.com/manbehindthemadness/describe/assets/24589462/23498605-2f54-4826-bcb1-b5bb19ed9a7f">

### Removing Text

To remove text entirely from an image, you can use the following code:
- Ensure to set `lama_refine=True` in the `Remover` initialization to utilize the `passes` parameter effectively for better text removal results.

```python
from unscribe import Remover
import cv2

# Initialize the Remover with debug and visualization enabled
remover = Remover(
    show_mats=True,
    debug=True,
    lama_refine=True
)

# Define parameters for removing text
image_path = "path/to/your/image.jpg"
low_clamp = 0.1
high_clamp = 0.9
mode = "remove"  # Set mode to "remove" for text removal
passes = 13

# Load the image
image = cv2.imread(image_path)

# Use the load_mat method to remove text from the image
removed_text_image = remover.load_mat(
    mat=image,
    low_clamp=low_clamp,
    high_clamp=high_clamp,
    mode=mode,
    passes=passes
)

# Display or save the resulting image with removed text
cv2.imshow("Text Removed", removed_text_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
<img width="949" alt="create" src="https://github.com/manbehindthemadness/describe/assets/24589462/ceb25d12-3c77-49f8-bc91-21157eec4f3b">


## Notes
- LaMa: Large mat processing and inpainting [repository](https://github.com/advimman/lama)
- Modern Craft: Character Region Awareness For Text Detection [repository](https://github.com/manbehindthemadness/modern-craft)
- Official CRAFT-PyTorch [repository](https://github.com/clovaai/CRAFT-pytorch)
- I own none of the images used in this document, diagrams are from other repositories and examples are random from the internet.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
