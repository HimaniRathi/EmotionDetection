import getopt
import glob
import sys
import os
import cv2
import numpy as np

from detect_face import get_largest_face, detect_faces

datadir = "data/"


def delete_frames():
    mydir = datadir + "frames"
    filelist = [f for f in os.listdir(mydir)]
    # print(filelist)
    for f in filelist:
        n = int(f.split('_')[2][:-4])
        if n % 4 != 0:
            # print(n)
            os.remove(os.path.join(mydir, f))


def get_videos(subject="*"):
    video_files = glob.glob(datadir + "savee/AudioVisualClip/" + subject + "/*.avi")
    print(video_files)
    n = 0
    for video in video_files:
        temp = video.split("/")
        # print("asdf", (video, temp[3]))
        print(n)
        n = n + 1
        vid2frames(video, temp[3], temp[4][:-4])


def vid2frames(path=datadir + "savee/AudioVisualClip/DC/a1.avi", subject="DC", vid_label="a1"):
    # noinspection PyArgumentList
    cap = cv2.VideoCapture(path)
    # print(cap)
    n = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        # print(str(n) +"\r")
        if ret is True:

            # Display the resulting frame
            # cv2.imshow('Frame', frame)
            frame = get_largest_face(frame, detect_faces(cv2.CascadeClassifier('lbpcascade_frontalface.xml'),
                                                         frame))
            frame = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(datadir + 'frames/' + subject + '_' + vid_label + '_' + str(n).zfill(5) + '.png', frame)
            n = n + 1
            # print(n)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def translate_labels(l):
    label = []
    if 'a' == l:
        # Angry
        label = np.append(label, ([1, 0, 0, 0, 0, 0, 0]))
    elif 'd' == l:
        # Disgust
        label = np.append(label, ([0, 1, 0, 0, 0, 0, 0]))
    elif 'f' == l:
        # Fear
        label = np.append(label, ([0, 0, 1, 0, 0, 0, 0]))
    elif 'h' == l:
        # Happy
        label = np.append(label, ([0, 0, 0, 1, 0, 0, 0]))
    elif 'n' == l:
        # Neutral
        label = np.append(label, ([0, 0, 0, 0, 1, 0, 0]))
    elif 'sa' == l:
        # Sad
        label = np.append(label, ([0, 0, 0, 0, 0, 1, 0]))
    elif 'su' == l:
        # Surprise
        label = np.append(label, ([0, 0, 0, 0, 0, 0, 1]))
    return label


def to_numpy_array():
    files = sorted(glob.glob(datadir + "frames/*.png"))
    # print(files)
    x_data = []
    final = []
    label = []
    l = ""
    for myFile in files:
        temp = myFile.split("_")

        image = cv2.imread(myFile, cv2.IMREAD_GRAYSCALE)
        if int(temp[2][:-4]) == 0 and len(x_data) is not 0:
            # print(myFile)
            l = ''.join([i for i in temp[1] if not i.isdigit()])
            label.append(translate_labels(l))
            # print(np.array(x_data).shape)
            fill = np.zeros((abs(60-np.array(x_data).shape[0]), 48, 48))
            final.append(np.concatenate((x_data, fill), axis=0)[0:60])
            x_data = []

        x_data.append(image)
        del image

    fill = np.zeros((abs(60 - np.array(x_data).shape[0]), 48, 48))
    final.append(np.concatenate((x_data, fill), axis=0)[0:60])
    del x_data
    video = np.array(final)
    del final
    label.append(translate_labels(l))
    video = video.reshape(-1, video.shape[1], video.shape[3], video.shape[2], 1)
    label = np.array(label)
    print("video", video.shape)
    print("label", label.shape)
    np.save(datadir + 'x_train', video)
    np.save(datadir + 'y_train', label)


def main(argv):
    global opts
    global datadir
    help_text = """data_cleaning.py  -h --help
    -h --help       Get help.
    -s --subject    Specify subject name[JK,DC,...]
    -d --delete     Delete every frame other than multiples of 4.
    -n --numpy      Save data/frames/ content as a numpy array in data/x_train.npy and data/y_train.npy
    -t --test       Test mode.(place first)
    """
    try:
        opts, args = getopt.getopt(argv, "hs:dnt")
    except getopt.GetoptError:
        print(help_text)
        sys.exit()
    subject = "*"
    for opt, arg in opts:
        if opt in ('-t', '--test'):
            datadir = "datat/"
        if opt in ('-h', '--help'):
            print(help_text)
            sys.exit()
        elif opt in ('-s', '--subject'):
            subject = str(arg)
            get_videos(subject)
            sys.exit()
        elif opt in ('-d', '--delete'):
            delete_frames()
            sys.exit()
        elif opt in ('-n', '--numpy'):
            to_numpy_array()
            sys.exit()
    print(help_text)


if __name__ == "__main__":
    # sys.stdout = open('logs/cleaning.txt', 'w')
    # sys.stderr = open('logs/cleaning-err.txt', 'w')
    main(sys.argv[1:])
