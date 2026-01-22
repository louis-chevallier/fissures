from utillc import *
import cv2, os, sys
import numpy as np
import tkinter as tk  
from PIL import Image,ImageTk  

folder = "/home/louis/Desktop/tmp/fissures"
imaf = "IMG_20251107_113638.jpg"
imbf = "IMG_20250328_170711.jpg"

imbf = imaf

imf = [ imaf, imbf]

ims_hide = [ cv2.imread(os.path.join(folder, i)) for i in imf]
topil = lambda x :Image.fromarray(np.uint8(x)).convert('RGB')

def new_size(img, base_width=500) :
	wpercent = (base_width / float(img.size[0]))
	hsize = int((float(img.size[1]) * float(wpercent)))
	return base_width, hsize

rsz = lambda x, sz : x.resize(sz, Image.Resampling.LANCZOS)


pims = [ topil(e) for e in ims_hide]
pims = [ rsz(e, new_size(e)) for e in pims]

EKOX(pims[0].size)
nw, nh = pims[0].size


EKOX(pims[0])


root = tk.Tk()  
root.title("fissures")

root.geometry("1000x1000")



##############"

# Convert images to grayscale
img1, img2 = [ cv2.cvtColor(np.array(im), cv2.COLOR_BGR2GRAY) for im in pims]
EKOX(img1.shape)

mean_img1 = np.mean(img1)
mean_img2 = np.mean(img2)

mean_img1, mean_img2 = [ np.mean(e) for e in (img1, img2) ]

# Calculate the ratio of the brightness of the images.
ratio = mean_img1 / mean_img2
print(f'Brightness ratio: {ratio}')

# Multiply the second image by the ratio.
img2 = img2 * ratio

# Convert the image to uint8 again.
img2 = np.clip(img2, 0, 255)
img2 = img2.astype(np.uint8)

EKOX(img1.shape)

image1 = img1 #cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
image2 = img2 #cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
height, width = image2.shape




cv = tk.Canvas()
cv.pack(side='top', fill='both', expand='yes')  


def save() :
	a_points = np.asarray([ center_circle(c) for c,_,_ in circles[::2]])
	with open('points.npy', 'wb') as f:
		np.save(f, a_points)

def load() :
	with open('points.npy', 'rb') as f:
		a = np.load(f)
	EKOX(a)
	[ do_new(cx, cy) for (cx, cy) in a]


def match() :
	a_points = np.asarray([ center_circle(c) for c,_,_ in circles[::2]])
	b_points = np.asarray([ center_circle(c) for c,_,_ in circles[1::2]])
	b_points = b_points - [ nw , 0]
	EKOX(a_points)
	EKOX(b_points)
	M, mask = cv2.findHomography(a_points, b_points, cv2.RANSAC, 5.0)
	EKOX(M.shape)
	EKOX(M)
	#draw_params = dict(singlePointColor=None, flags=2)
	#img3 = cv2.drawMatches(image1, a_points, image2, b_points)
	#EKOI(img3)
	aligned_image2 = cv2.warpPerspective(image2, M, (image1.shape[1], image1.shape[0]))
	EKOI(aligned_image2)
	


photo1, photo2 = [ ImageTk.PhotoImage(e) for e in pims]
cv.create_image(0, 0, image=photo1, anchor='nw') 
cv.create_image(photo1.width(), 0, image=photo2, anchor='nw')


marks = []

radius_circle = 30

def coords4(center, radius=30) :
	cx, cy = center
	return [cx - radius,
			cy - radius,
			cx + radius,
			cy + radius]


def add_two_marks(c) :
	cx, cy = c
	n = numbers.pop(0)
	a = add_mark((cx, cy), "A" + str(n))
	b = add_mark((cx + 10, cy), "B" + str(n))
	return a, b
	
def add_mark(c, ss) :
	x, y, _, _ = coord = coords4(c)
	circle = cv.create_oval(coord, outline="red", width=2)
	text = cv.create_text((x, y), text=ss, font="Times 20 italic bold", fill="red")
	dot = cv.create_oval(coords4(c, 2), fill="green", outline="red", width=2)
	return circle, text, dot

N=1

numbers = list(range(1222))

cc = np.random.randint(0, nw, size=(N, 2))
EKOX(cc)
	
circles = [ e for i, c in enumerate(cc) for e in add_two_marks(c) ]

selected = None
click_down_coords = None

def center_circle(circle) :
	x, y, _, _ = cv.coords(circle)
	return x + radius_circle, y + radius_circle

floats = lambda l : np.asarray(list(map(float, l)))

threshold = 4

def do_new(cx, cy) :
	global selected		
	newn = numbers.pop(0)
	new_marka = add_mark((cx, cy), "A" + str(newn))
	new_markb = add_mark((cx+10, cy), "B" + str(newn))
	circles.append(new_marka)
	circles.append(new_markb)
	selected = len(circles)-2
	#EKOX(newn)

def click(c):
	global selected, click_down_coords
	cx, cy = c.x, c.y
	c = cx, cy
	click_down_coords = c
	
	if len(circles) == 0 :
		do_new(cx, cy)
	else :
		dists = [ np.linalg.norm(floats(center_circle(e)) - floats(c)) for e,t,d in circles] # coin bas gauche
		mm = np.argmin(dists)	
		if dists[mm] < threshold :
			selected = mm
			#EKOX(selected)
		else :
			do_new(cx, cy)
	
def drag(c):
	cx, cy = c.x, c.y
	if selected is not None :
		mm = selected
		sel_circle, sel_text, sel_dot = circles[mm]
		#EKOX(sel_circle)
		ccc = coords4((cx, cy))
		#EKOX(ccc)
		x1, y1, x2, y2 = ccc
		cv.coords(sel_circle, x1, y1, x2, y2)
		cv.coords(sel_text, x1, y1)
		ccc = coords4((cx, cy), 2)
		x1, y1, x2, y2 = ccc
		cv.coords(sel_dot, x1, y1, x2, y2)
		#EKOX(center_circle(sel_circle))


def close(ca, cb) :
	return np.linalg.norm(floats(ca) - floats(cb)) < threshold 


def delete(idx) :
	oo = circles[idx]
	[ cv.delete(e) for e in oo]

	
def release(c):
	global selected
	cx, cy = c.x, c.y
	c = cx, cy
	dists = [ np.linalg.norm(floats(center_circle(e)) - floats(c)) for e,t,d in circles] # coin bas gauche
	mm = np.argmin(dists)
	if dists[mm] < threshold :
		selected = mm
		#EKOX(selected//2)
		if close(c, click_down_coords) :
			to_kill = selected//2*2
			EKOX(to_kill)
			delete(to_kill)
			delete(to_kill+1)
			del circles[to_kill]
			del circles[to_kill]


def delete_all() :
	EKO()
	global circles
	[ delete(i) for i in range(len(circles))]
	circles = []

	
def key_handler(event) :
	EKOX(event.keysym)
	try :
		{
			'D' : delete_all,
			'q' : lambda : sys.exit(0),
			'l' : load,
			's' : save,
			'm' : match
		}[event.keysym]()
		EKO()
	except Exception as ex:
		EKOX(ex)
		pass

			
cv.bind('<ButtonPress-1>', click)
cv.bind('<ButtonRelease-1>', release)
cv.bind("<B1-Motion>", drag) 
root.bind('<KeyPress>', key_handler)
root.bind('<Shift-d>', key_handler)
EKO()


root.mainloop()



# Source - https://stackoverflow.com/q/76568361
# Posted by Daniel Agam, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-21, License - CC BY-SA 4.0


EKOX(ims[0].shape)

# Convert images to grayscale
img1, img2 = [ cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in ims]
EKOX(img1.shape)

# crop the images to ROI - The ROI is the same for both images and is the left bottom corner in size of 1/5 of the image
img1_roi = img1[0:int(img1.shape[0] / 5), 0:int(img1.shape[1] / 5)]
img2_roi = img2[0:int(img2.shape[0] / 5), 0:int(img2.shape[1] / 5)]

img1_roi, img2_roi = [ e[0:int(e.shape[0] / 5), 0:int(e.shape[1] / 5)] for e in [ img1_roi, img2_roi]]


# Calculate the mean of the images.
mean_img1 = np.mean(img1_roi)
mean_img2 = np.mean(img2_roi)

mean_img1, mean_img2 = [ np.mean(e) for e in (img1_roi, img2_roi) ]

# Calculate the ratio of the brightness of the images.
ratio = mean_img1 / mean_img2
print(f'Brightness ratio: {ratio}')

# Multiply the second image by the ratio.
img2 = img2 * ratio

# Convert the image to uint8 again.
img2 = np.clip(img2, 0, 255)
img2 = img2.astype(np.uint8)


EKOX(img1.shape)


# Source - https://stackoverflow.com/q/76568361
# Posted by Daniel Agam, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-21, License - CC BY-SA 4.0

image1 = img1 #cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
image2 = img2 #cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
height, width = image2.shape

sift = cv2.SIFT_create()
EKO()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
EKO()
bf = cv2.BFMatcher()

matches = bf.knnMatch(descriptors1, descriptors2, k=2)
EKOX(len(matches))
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
EKOX(M.shape)

draw_params = dict(singlePointColor=None, flags=2)
img3 = cv2.drawMatches(image1, keypoints1, image2, keypoints2)
EKOI(img3)

# Source - https://stackoverflow.com/q/76568361
# Posted by Daniel Agam, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-21, License - CC BY-SA 4.0



aligned_image = cv2.warpAffine(image2, M[0:2, :], (image1.shape[1], image1.shape[0]))
aligned_image2 = cv2.warpPerspective(image2, M, (image1.shape[1], image1.shape[0]))

EKOI(aligned_image)
EKOI(aligned_image2)
