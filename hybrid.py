import numpy as np
import math
import cv2
from scipy.stats import multivariate_normal
from scipy import ndimage

##
##	NOTE: For some reason it doesn't show very well when it pops the window up, but it looks right if you check the output file. 
##	I really have no idea why this happens so I just turned off the preview window
##
def hybrid(img0, img1):
	
	lo_pass = lowpass(img0)
	cv2.imwrite("lo_pass.png", lo_pass)
	hi_pass = highpass(img1)
	cv2.imwrite("hi_pass.png", hi_pass)

	return morph(hi_pass, lo_pass)
	
#breaks the image into channels and then applies the low pass filter to them before recombining
def lowpass(img):
	b,g,r = cv2.split(img)
	b_ = convolve(b, generateGaussianKernel(sigma))
	g_ = convolve(g, generateGaussianKernel(sigma))
	r_ = convolve(r, generateGaussianKernel(sigma))
	return cv2.merge([b_,g_,r_])
	
#breaks up the channels, finds their low pass frequencies, subtracts those frequencies from the main
#image and then sharpens it a bit
def highpass(img):
	b,g,r = cv2.split(img)
	b2,g2,r2 = cv2.split(lowpass(img))	#get the low pass filter so we can subtract it 
	sharpKern = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
	print(type(b[0][0]))
	#sharpen the channels and then subtract that from the lowpass image
	b_ = mySub(convolve(b, sharpKern),b2)
	g_ = mySub(convolve(g, sharpKern),g2)
	r_ = mySub(convolve(r, sharpKern),r2)
	print(type(b_[0][0]))
	return cv2.merge([b_,g_,r_])
	
#my implementation of the convolution function
def convolve(image, kernel):
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")
	for y in range(pad, iH + pad):
		for x in range(pad, iW + pad):
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
			k = (roi * kernel).sum()
			output[y - pad, x - pad] = k
	return output
	
#Where the high and low pass inputs are split and added together
def morph(hi_pass, lo_pass):
	hb,hg,hr = cv2.split(hi_pass)
	lb,lg,lr = cv2.split(lo_pass)
	
	b = myAdd(hb, lb)
	g = myAdd(hg, lg)
	r = myAdd(hr, lr)
	
	return cv2.merge([b,g,r])
	
#my implementation of a function that adds the pixel values of 2 channels together.
def myAdd(img0, img1):
	output = np.zeros((img0.shape[0], img0.shape[1]))
	print(output.shape)
	for i in range(img0.shape[0]):
		for j in range(img0.shape[1]):
			output[i][j] = img0[i][j]+img1[i][j]
	return output

#same as above, except now I'm subtracting. 
def mySub(img0, img1):
	output = np.zeros((img0.shape[0], img0.shape[1]))
	for i in range(img0.shape[0]):
		for j in range(img0.shape[1]):
			output[i][j] = img0[i][j] - img1[i][j]
	return output
	
#generates a gaussian kernal using sigma as a seed for the kernel size  
def generateGaussianKernel(sigma):
	size = 2 * math.ceil(3*sigma)+1
	x, y = np.mgrid[-size:size+1, -size:size+1]
	normal = 1 / (2.0 * np.pi * sigma**2)
	g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
	return g

	
sigma = 8
output = hybrid(cv2.imread('left.jpg',1),cv2.imread('right.jpg',1))

cv2.imwrite ('hybrid.png', output)
print("Finished and file saved")
cv2.waitKey(0)
cv2.destroyAllWindows()