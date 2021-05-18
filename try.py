from ITTISaliencyLib import get_spatial_saliency_map
import cv2
import matplotlib.pyplot as plt

image_test = cv2.imread("./examples/MUG.jpg")
saliency_map = get_spatial_saliency_map(image_test)

plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB))
plt.title('Nice mug')
plt.subplot(1, 2, 2), plt.imshow(saliency_map, 'gray')
plt.title('Saliency mug')
plt.show()