import numpy as np
import cv2

#Modification de l'algorithme SIFT permettant d'améliorer légerement les performances.
#On normalise les descripteurs, puis on prend leur racine carrés pour avoir la norme L2
class RootSIFT:
	def __init__(self):
		self.extractor = cv2.SIFT_create()

	def compute(self, image, kps, eps = 1e-7):
		(kps, descs) = self.extractor.compute(image, kps)
		if len(kps) == 0:
			return([], None)

		descs /= (descs.sum(axis=1, keepdims = True)+eps)
		descs = np.sqrt(descs)

		return (kps, descs)

