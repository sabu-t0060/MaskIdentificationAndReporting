import cv2
import sys

class FeatureExtract:

    def __init__(self):
        pass

    def extract_face(self,infile,outdir,pos):

        imagePath = infile

        #sys.argv[1]

        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
            )

        print("[INFO] Found {0} Faces.".format(len(faces)))

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = image[y:y + h, x:x + w]
            print("[INFO] Object found. Saving locally.")
            cv2.imwrite(f'{outdir}/{pos}_faces.jpg', roi_color)

        #status = cv2.imwrite(f'{outdir}/faces_detected.jpg', image)
        #print("[INFO] Image faces_detected.jpg written to filesystem: ", status)


    

 # it uses a haarcascades classifier to dectect whether the image contains the face
    def check_face(self,infile):

        imagePath = infile

        #sys.argv[1]
         #first reads image from the imagepath
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       # it now checks the image using  haarcascades classifier 
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
            )

        print("[INFO] Found {0} Faces.".format(len(faces)))
        return len(faces) #sends number of faces found 
        
 # uses a haarcascades classifier to dectect whether the image contains the eyes
    def check_eye(self, infile):

        imagePath = infile

        # sys.argv[1]
         #first reads image from the imagepath
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       # it now checks using  haarcascades classifier 
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        eyes = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )

        print("[INFO] Found {0} Eyes.".format(len(eyes)))

        # status = cv2.imwrite(f'{outdir}/faces_detected.jpg', image)
        # print("[INFO] Image faces_detected.jpg written to filesystem: ", status)
        return len(eyes) #sends number of eyes found