import numpy as np
import cv2

def rotate_image(image: np.ndarray, angle: float, pad: bool = True) -> np.ndarray:
    """
    Rotate image by angle degrees.
    If pad=True -> keep all pixels visible by expanding canvas and fill empty area with black.
    If pad=False -> warp to original image size (may crop/zoom).
    """
    h, w = image.shape[:2]
    center = (w/2.0, h/2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    if pad:
        cos = abs(M[0,0])
        sin = abs(M[0,1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0,2] += (new_w/2) - center[0]
        M[1,2] += (new_h/2) - center[1]
        rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
        return rotated
    else:
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
        return rotated
    

def rotate_points_with_pad(points, angle, orig_w, orig_h):
    """
    Rotál centroid pontokat pad-es rotált kép koordinátáiba.
    points: lista pl. [(1473,935), (1502,816)]
    angle: fokban
    orig_w, orig_h: eredeti frame méret
    """
    # Eredeti kép közepe
    cx, cy = orig_w / 2, orig_h / 2

    # Eredeti rotációs mátrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # Új (pad-elt) szélesség, magasság
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(orig_h * sin + orig_w * cos)
    new_h = int(orig_w * sin + orig_h * cos)

    # Pad miatti eltolás
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    # Kimeneti pontok tömbbe
    pts = np.array(points, dtype="float32")

    # Homogén koordináták
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])

    # Mátrixszorzás
    rotated = pts_h @ M.T

    return [(float(x), float(y)) for x, y in rotated]
    

def fetch_roi_img(img, orig_width, orig_height):
    img_height, img_width = img.shape[:2]

    h_pad = int((img_height - orig_height) / 2)
    w_pad = int((img_width - orig_width) / 2)

    if h_pad > 0 and w_pad > 0:
        img = img[h_pad : -h_pad, w_pad : -w_pad]
    return img


def add_grid(img, step=500, small_color=(0, 127, 0), big_color=(0, 255, 0)):
    h, w = img.shape[:2]
    # Vékony rácsvonalak (kis négyzetek)
    for x in range(0, w, step//5):
        cv2.line(img, (x, 0), (x, h), small_color, max(1, step//150))
    for y in range(0, h, step//5):
        cv2.line(img, (0, y), (w, y), small_color, max(1, step//150))
    # Vastagabb vonalak (nagy négyzetek)
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), big_color, max(1, step//100))
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), big_color, max(1, step//100))

    return img

def add_dynamic_grid(
    img,
    activity_level=0.5,
    step=140
):

    overlay = img.copy()

    h, w = img.shape[:2]

    intensity = int(
        255 * activity_level
    )

    for x in range(0, w, step):

        cv2.line(
            overlay,
            (x, 0),
            (x, h),
            (0, intensity, 0),
            1
        )

    for y in range(0, h, step):

        cv2.line(
            overlay,
            (0, y),
            (w, y),
            (0, intensity, 0),
            1
        )

    cv2.addWeighted(
        overlay,
        0.3,
        img,
        0.7,
        0,
        img
    )

    return img

if __name__ == "__main__":
    height = 800
    width = 800
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    image = add_grid(img)

    # Eredmény mentése
    cv2.imwrite("grid_overlay.png", image)

    # Megjelenítés
    cv2.imshow("Grid", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()