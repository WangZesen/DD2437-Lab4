import matplotlib.pyplot as plt
import numpy as np

def show_image(data):
	img = np.array(data)
	img = np.reshape(data, (28, 28)).tolist()
	plt.imshow(img)
	plt.show()
