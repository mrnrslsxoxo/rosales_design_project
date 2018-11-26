import numpy as np
import cv2
import dlib
from scipy.spatial import distance as dist
from scipy.spatial import ConvexHull

#--------------------------------------------------------------------------

# Setting-up the parameters for the dlib Face Detector and Face Landmark Detector
# Initialize the arrays that extracts the individual face landmarks out of 68 landmarks which Dlib returns

PREDICTOR_PATH = "resources/landmark_predictor.dat"

FULL_POINTS = list(range(0, 68))
FACE_POINTS = list(range(17, 68))

JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(PREDICTOR_PATH)

#--------------------------------------------------------------------------

# Function for measuring the size and locating the center of face parts

def part_size(part, from_point, to_point):
	partWidth = dist.euclidean(part[from_point], part[to_point])
	hull = ConvexHull(part)
	partCenter = np.mean(part[hull.vertices, :], axis=0)
	partCenter = partCenter.astype(int)
	return int(partWidth), partCenter

#--------------------------------------------------------------------------

# Function for placing the overlay on to the face image

def place_part(frame, partCenter, partSize, multiplier, imgPart):
	partSize = int(partSize * multiplier)

	x1 = int(partCenter[0,0] - (partSize/2))
	x2 = int(partCenter[0,0] + (partSize/2))
	y1 = int(partCenter[0,1] - (partSize/2))
	y2 = int(partCenter[0,1] + (partSize/2))

	h, w = frame.shape[:2]

	# Check for clipping
	if x1 < 0:
		x1 = 0
	if y1 < 0:
		y1 = 0
	if x2 > w:
		x2 = w
	if y2 > h:
		y2 = h

	# Re-calculate the size to avoid clipping
	partOverlayWidth = x2 - x1
	partOverlayHeight = y2 - y1

	# Calculate the masks for the overlay
	partOverlay = cv2.resize(imgPart["imgPart"], (partOverlayWidth,partOverlayHeight), interpolation = cv2.INTER_AREA)
	mask = cv2.resize(imgPart["orig_mask"], (partOverlayWidth,partOverlayHeight), interpolation = cv2.INTER_AREA)
	mask_inv = cv2.resize(imgPart["orig_mask_inv"], (partOverlayWidth,partOverlayHeight), interpolation = cv2.INTER_AREA)

	# Take ROI for the overlay from background, equal to size of the overlay image
	roi = frame[y1:y2, x1:x2]

	# Contains the original image only where the overlay is not in the region that is the size of the overlay
	roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

	# Contains the image pixels of the overlay only where the overlay should be
	roi_fg = cv2.bitwise_and(partOverlay,partOverlay,mask = mask)

	# Join the roi_bg and roi_fg
	dst = cv2.add(roi_bg,roi_fg)

	# Place the joined image, saved to dst back over the original image
	frame[y1:y2, x1:x2] = dst

#--------------------------------------------------------------------------

# Function for image load and pre-process

def filter_image(location):
	# Load the image to be used as overlay
	imgPart = cv2.imread(location,-1)
	# Create the mask from image
	orig_mask = imgPart[:,:,3]
	# Create the inverted mask from the overlay image
	orig_mask_inv = cv2.bitwise_not(orig_mask)
	# Convert the overlay image to BGR
	imgPart = imgPart[:,:,0:3]
	# Save the original image size
	origPartHeight, origPartWidth = imgPart.shape[:2]

	# Return the parameters processed
	return {"imgPart":imgPart, "orig_mask":orig_mask, "orig_mask_inv":orig_mask_inv, "origPartHeight":origPartHeight, "origPartWidth":origPartWidth}

#--------------------------------------------------------------------------

# Specify location of image to be used as overlay
# Call-out function 'filter_image' to load and pre-process the images

eye_location = "resources/eye_filter.png"
eye_filter = filter_image(eye_location)

lips_location = "resources/lips_filter.png"
lips_filter = filter_image(lips_location)

text_location = "resources/text_filter.png"
text_filter = filter_image(text_location)

head_location = "resources/head_filter.png"
head_filter = filter_image(head_location)

glasses_location = "resources/sunglass_filter.png"
glasses_filter = filter_image(glasses_location)

mustache_location = "resources/mustache_filter.png"
mustache_filter = filter_image(mustache_location)

#--------------------------------------------------------------------------

# Start capturing the webcam
video_capture = cv2.VideoCapture(0)

filter_number = int(input("1 or 2? "))

if filter_number == 1:
	while True:
		ret, frame = video_capture.read()

		if ret:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			rects = detector(gray, 0)

			for rect in rects:
				x = rect.left()
				y = rect.top()
				x1 = rect.right()
				y1 = rect.bottom()

				landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])

				left_eye = landmarks[LEFT_EYE_POINTS]
				right_eye = landmarks[RIGHT_EYE_POINTS]
				outline_mouth = landmarks[MOUTH_OUTLINE_POINTS]
				outline_jaw = landmarks[JAWLINE_POINTS]

				leftEyeSize, leftEyeCenter = part_size(left_eye, 0, 3)
				rightEyeSize, rightEyeCenter = part_size(right_eye, 0, 3)
				outlineLipsSize, outlineLipsCenter = part_size(outline_mouth, 0, 6)
				outlineMouthSize, outlineMouthCenter = part_size(outline_mouth, 3, 9)
				outlineJawSize, outlineJawCenter = part_size(outline_jaw, 0, 16)

				place_part(frame, leftEyeCenter, leftEyeSize, 2.0, eye_filter)
				place_part(frame, rightEyeCenter, rightEyeSize, 2.0, eye_filter)
				place_part(frame, outlineJawCenter - 200, outlineJawSize * 0.5, 2.0, text_filter)

				if outlineLipsSize < 65 and outlineMouthSize < 35:
					place_part(frame, outlineLipsCenter, outlineLipsSize, 1.0, lips_filter)

			cv2.imshow("Filter", frame)

		ch = 0xFF & cv2.waitKey(1)

		if ch == ord('q'):
			break

if filter_number == 2:
	while True:
		ret, frame = video_capture.read()

		if ret:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			rects = detector(gray, 0)

			for rect in rects:
				x = rect.left()
				y = rect.top()
				x1 = rect.right()
				y1 = rect.bottom()

				landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])

				middle_nose = landmarks[NOSE_POINTS]
				outline_mouth = landmarks[MOUTH_OUTLINE_POINTS]
				outline_jaw = landmarks[JAWLINE_POINTS]

				outlineMouthSize, outlineMouthCenter = part_size(outline_mouth, 2, 4)
				outlineJawSize, outlineJawCenter = part_size(outline_jaw, 0, 16)

				place_part(frame, outlineMouthCenter, outlineMouthSize, 6.0, mustache_filter)
				place_part(frame, outlineJawCenter - 200, outlineJawSize * 0.5, 2.0, head_filter)

			cv2.imshow("Filter", frame)

		ch = 0xFF & cv2.waitKey(1)

		if ch == ord('q'):
			break

cv2.destroyAllWindows()
