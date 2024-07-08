from unscribe import Remover
import cv2

# Initialize the Remover with debug and visualization enabled
remover = Remover(
    show_mats=True,
    debug=True,
    lama_refine=True,
    gpu='cuda'
)

# Define parameters for removing text
image_path = "test.jpg"
low_clamp = 0.1
high_clamp = 0.9
mode = "scramble"  # Set mode to "remove" for text removal
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
