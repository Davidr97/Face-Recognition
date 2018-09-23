import dlib


# Face detection
def face_detection(img):

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should up-sample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, 1)

    return faces
