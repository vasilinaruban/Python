import cv2
import numpy as np

image = cv2.imread('lab7.png')
ghost_images = ['candy_ghost.png', 'pampkin_ghost.png', 'scary_ghost.png']
sift = cv2.SIFT_create()
flann = cv2.FlannBasedMatcher()

for ghost_image_path in ghost_images:
    ghost_image = cv2.imread(ghost_image_path, cv2.IMREAD_GRAYSCALE)
    if ghost_image is not None:
        kp1, des1 = sift.detectAndCompute(ghost_image, None)
        kp2, des2 = sift.detectAndCompute(image, None)

        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            h, w = ghost_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            ghost_corners = cv2.perspectiveTransform(pts, M)

            image = cv2.polylines(image, [np.int32(ghost_corners)], isClosed=True, color=(0, 255, 0), thickness=2)

cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
