import dlib


# Face and points detection
def face_points_detection(img, bbox):
    predictor_path = 'models/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, bbox)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    cords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

    # return the list of (x, y)-coordinates
    return cords

