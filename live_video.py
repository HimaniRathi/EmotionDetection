import getopt
import numpy as np
import sys
import cv2
from keras import Model

from keras.models import model_from_json
from keras.utils import plot_model

from detect_face import get_largest_face, detect_faces

model = ""

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

datadir = 'data/'
shape = (48, 48)
shapec = (1, 48, 48, 1)


json_file = open(datadir + 'ckplus.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ckplus = model_from_json(loaded_model_json)

# load weights into new model
ckplus.load_weights(datadir + 'ckplus.h5')
# ckplus.summary()

layer_name = 'flatten_1'
intermediate_layer_model = Model(input=ckplus.input, output=ckplus.get_layer(layer_name).output)


def predict_emotion_image(face_image):
    if not shapec[-1] == 1:
        face_image = (np.moveaxis(face_image, -1, 0)).reshape(shapec)
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray, shape, interpolation=cv2.INTER_AREA)
    image = resized_img.reshape(shapec)
    # print(image.shape)
    list_of_list = model.predict(image, batch_size=1, verbose=1)
    print(list_of_list)
    angry, disgust, fear, happy, neutral, sad, surprise = [prob for lst in list_of_list for prob in lst]
    return [angry, disgust, fear, happy, neutral, sad, surprise]


def predict_emotion_video(face_video):
    face_video = np.copy(face_video).reshape(shapec)
    list_of_list = model.predict(face_video, batch_size=1, verbose=1)
    print(list_of_list)
    angry, disgust, fear, happy, neutral, sad, surprise = [prob for lst in list_of_list for prob in lst]
    return [angry, disgust, fear, happy, neutral, sad, surprise]


def put_emoji(angry, disgust, fear, happy, neutral, sad, surprise):
    emotion = max(angry, disgust, fear, happy, neutral, sad, surprise)
    if emotion == angry:
        status = "angry"
        # emoji = cv2.imread("../data/angry.png")
        # print(" You are angry")
    elif emotion == fear:
        status = "fear"
        # emoji = cv2.imread("../data/fearful.png")
        # print("fear")
    elif emotion == happy:
        status = "happy"
        # emoji = cv2.imread("../data/happy.png")
        # print(" You are happy")
    elif emotion == sad:
        status = "sad"
        # emoji = cv2.imread("../data/sad.png")
        # print(" You are sad")
    elif emotion == surprise:
        status = "surprise"
        # emoji = cv2.imread("../data/surprised.png")
        # print(" You are surprise")
    elif emotion == disgust:
        status = "disgust"
    else:
        # emotion == neutral:
        status = "neutral"
        # emoji = cv2.imread("../data/neutral.png")
        # print(" You are neutral")
    # emoji = cv2.resize(emoji, (120, 120), interpolation=cv2.INTER_CUBIC)
    # overlay = cv2.resize(emoji, (80, 80), interpolation=cv2.INTER_AREA)
    return status


def video_capture(image_based=False):
    cv2.namedWindow("exit on ESC")
    # to capture video from cv2
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, (640,
    vc = cv2.VideoCapture(0)
    frame = 0
    video = []
    temp_video = []
    fill = np.zeros(shape)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    f = cv2.flip(frame, 1)
    frames_passed = 0
    frame_rate = 2
    plot = frame[-120:, 0:120]
    (angry, disgust, fear, happy, neutral, sad, surprise) = (0, 0, 0, 0, 0, 0, 0)
    while rval:
        cv2.imshow("exit on ESC", f)
        rval, frame = vc.read()
        # tilt optimization req
        # temp = crop_rot_images(frame, lbp_face_cascade,draw_face=True)
        # if frames_passed % frame_rate == 0:
        temp = get_largest_face(frame, detect_faces(cv2.CascadeClassifier('lbpcascade_frontalface.xml'), frame),
                                draw_face=True)
        # temp = draw_faces(frame, detect_faces(lbp_face_cascade, frame))

        if temp.shape != (0, 0, 3):

            frame = cv2.flip(frame, 1)
            if image_based:
                angry, disgust, fear, happy, neutral, sad, surprise = predict_emotion_image(temp)
            else:
                # implemented for only transfer learning part
                if not frames_passed % frame_rate == 3:
                    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
                    temp = cv2.resize(temp, (48, 48), interpolation=cv2.INTER_AREA)
                    resized = (np.moveaxis(temp, -1, 0)).reshape((1, 1, 48, 48))
                    # print(resized.shape)
                    vector = intermediate_layer_model.predict(resized)
                    # print(vector.shape)

                    if len(video) >= 60:
                        video.append(vector.reshape(4608))
                        temp_video = video[1:61]
                    else:
                        video.append(vector.reshape(4608))
                        temp_video = np.concatenate((video, fill), axis=0)[0:60]
                    # out.write(frame)
                angry, disgust, fear, happy, neutral, sad, surprise = predict_emotion_video(temp_video)
                frames_passed = frames_passed + 1
            print(put_emoji(angry, disgust, fear, happy, neutral, sad, surprise))
            # cv2.putText(frame, status, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

            # with open('emotion.txt', 'a') as fp:
            #     fp.write('{},{},{},{},{},{},{}\n'.format(time.time(), angry, fear, happy, sad, surprise, neutral))

            f = frame
        else:
            f = cv2.flip(frame, 1)

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    vc.release()
    cv2.destroyWindow("exit on ESC")
    # out.release()


def main(argv):
    global opts
    global model
    global shape
    global shapec
    help_text = """live_video.py  -h --help
    -h --help       Get help.
    -m --model      Model number
    """
    try:
        opts, args = getopt.getopt(argv, "hm:")
    except getopt.GetoptError:
        print(help_text)
        sys.exit()
    modelno = 0
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(help_text)
            sys.exit()
        elif opt in ('-m', '--model'):
            modelno = int(arg)
    if modelno == 0:
        shape = (48, 48)
        shapec = (1, 48, 48, 1)
        # load json and create model arch
        json_file = open('data/modelimagebased.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # load weights into new model
        model.load_weights('data/modelimagebased.h5')
        model.summary()
        print(model.input_shape)
        video_capture(image_based=True)
    if modelno == -1:
        shape = (48, 48)
        shapec = (1, 1, 48, 48)
        # load json and create model arch
        json_file = open('data/ckplus.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # load weights into new model
        model.load_weights('data/ckplus.h5')
        model.summary()
        print(model.input_shape)
        video_capture(image_based=True)
    if modelno == 1:
        shape = (60, 4608)
        shapec = (1, 60, 4608)
        json_file = open('data/modeltransferA.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # load weights into new model
        model.load_weights('data/modeltransferA.h5')
        model.summary()
        # plot_model(ckplus, to_file='ck.svg', show_shapes=True, show_layer_names=True, rankdir='LR')
        # plot_model(model, to_file='gru.svg', show_shapes=True, show_layer_names=True, rankdir='LR')
        print(model.input_shape)
        video_capture()


if __name__ == "__main__":
    # sys.stdout = open('logs/cleaning.txt', 'w')
    # sys.stderr = open('logs/cleaning-err.txt', 'w')
    main(sys.argv[1:])
