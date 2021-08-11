#1. Added Thread
#2. Added camera capture
#3. Added Opencv face extract
#4. Email sending added
#https://www.google.com/settings/security/lesssecureapps
import cv2
import threading
import os
import shutil
import time

from feature_extract import FeatureExtract
from cnn import fetch_model, predict


import smtplib
import ssl
from email.mime.text import MIMEText
from email.utils import formataddr
from email.mime.multipart import MIMEMultipart  # New line
from email.mime.base import MIMEBase  # New line
from email import encoders  # New line


capture_file_path = 'capture'
face_file_path = 'face'
label = {0:'Mask',1:'No Mask'}

def initDir(file_path):
    shutil.rmtree(file_path, ignore_errors=True)
    os.mkdir(file_path);




def startCapture():
    initDir(capture_file_path)#set up the capture directory to save each caputed  image
    
    initDir(face_file_path)# set up the capture directory to save extracted face
    model = fetch_model('C:/finaltesting/data_set/') #fetch model by calling fetch_model

#get the webcam object and turn on the camera 
    fe = FeatureExtract()
    videoCaptureObject = cv2.VideoCapture(0)
    flag = True
    cnt = 0
    while(flag):
        ret,frame = videoCaptureObject.read()
        in_file = f'{capture_file_path}/camera_{cnt}.jpg' 
        cv2.imwrite(in_file,frame) #get the image from the video
        print('Captured>',f'{capture_file_path}/camera_{cnt}.jpg')
        
        #check for face in that image using check face fuction check_face() from the 
        #feature_extract.py in oder to check with the haarcascades classifier 
        face_cnt = fe.check_face(in_file) #

        if face_cnt == 0:  #if the result fails,then the system now  
                           #checks for eye using the function check_eye() from feature_extract.py
            eye_cnt = fe.check_eye(in_file)
            if eye_cnt == 0: # if the result fails agin ,then the
                             #the system cannot find any faces in that image
                print('No Face')
            else:            #if the result sucess,then the the system find the eyes, without other features of faces
                             #which probaly results to a face with a mask 
                          
                result = predict(in_file, model) #now the image checks with predict function from our model
                print('Mask', label[result])#now it display the result from both techiniques: ie the result using 
                                            #haarcascades classifier as mask and the result based on the 
                                            # Model which may be masked or Non masked

        else:                #if the classifiesr finds the image with features of faces 
            result = predict(in_file,model) # then its is checked with model in oder make ensure that its is Non Mask
            print('No Mask', label[result]) # result is displayed
            if label[result] == 'No Mask': # so if the image is predict with 'No Mask'
                fe.extract_face(in_file, face_file_path, cnt) #Extact the face from that iamge and save to face directory
                send_email(in_file) #send the email using the function
        time.sleep(1)
        if cnt == 100:
            flag = False
        cnt += 1
    time.sleep(1)
    videoCaptureObject.release()
    cv2.destroyAllWindows()

def send_email(filename):
  
    # User configuration : in user confifiguration we have to specify the sender and receiver emails
    sender_email = 'facedetect4495@gmail.com'
    sender_name = 'Mask No Mask'
    password = '*****'  # input('Please, type your password:n')

    receiver_emails = ['toshum95@gmail.com']
    receiver_names = ['Administrator']
    
    print("Sending the email...")

    mail_content = '''Hello,
    The following image is of a person not wearing the mask
    Thank You
    '''
    
    # set up the Email body specifying the From, to , subject , message and image attachment
    # email_html = open('email.html')
    email_body = mail_content  # email_html.read()

    #filename = './mask/camera_0.jpg'

    for receiver_email, receiver_name in zip(receiver_emails, receiver_names):
        print("Sending the email...")
        # Configurating user's info
        msg = MIMEMultipart()
        msg['To'] = formataddr((receiver_name, receiver_email))
        msg['From'] = formataddr((sender_name, sender_email))
        msg['Subject'] = 'Hello, my friend ' + receiver_name

        msg.attach(MIMEText(email_body, 'html'))

        try:
            # Open PDF file in binary mode
            with open(filename, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")  
                part.set_payload(attachment.read())

            # Encode file in ASCII characters to send by email
            encoders.encode_base64(part)

            # Add header as key/value pair to attachment part
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {'filename.jpg'}",
            )

            msg.attach(part)
        except Exception as e:
            print(f"Oh no! We didn't found the attachment!n{e}")
            break
# after the email is craeted open a SMTP connection and send the email
        try:
            # Creating a SMTP session | use 587 with TLS, 465 SSL and 25
            server = smtplib.SMTP('smtp.gmail.com', 587)
            # Encrypts the email
            context = ssl.create_default_context()
            server.starttls(context=context)
            # We log in into our Google account
            server.login(sender_email, password)
            # Sending email from sender, to receiver with the email body
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print('Email sent!')
        except Exception as e:
            print(f'Oh no! Something bad happened!n{e}')
            break
        finally:
            print('Closing the server...')
            server.quit()


if __name__ == "__main__":
    # creating thread
    #dataThread = threading.Thread(target=startDatasetCapture('nomask'), args=())
    # starting thread
    #dataThread.start()

    # creating thread
    captureThread = threading.Thread(target=startCapture(), args=())
    # starting thread
    captureThread.start()
    print("Done!")