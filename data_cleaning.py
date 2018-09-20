import getopt
import glob
import sys
import os
import cv2


def delete_frames():
    mydir = "data/frames"
    filelist = [f for f in os.listdir(mydir)]
    # print(filelist)
    for f in filelist:
        n = int(f.split('_')[2][:-4])
        if n % 4 != 0:
            # print(n)
            os.remove(os.path.join(mydir, f))


def get_videos(subject="*"):
    video_files = glob.glob("data/savee/AudioVisualClip/" + subject + "/*.avi")
    # print(video_files)
    n = 0
    for video in video_files:
        temp = video.split("/")
        # print("asdf", (video, temp[3]))
        print(n)
        n = n + 1
        vid2frames(video, temp[3], temp[4][:-4])


def vid2frames(path="data/savee/AudioVisualClip/DC/a1.avi", subject="DC",vid_label="a1"):
    cap = cv2.VideoCapture(path)
    # print(cap)
    n = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        # print(str(n) +"\r")
        if ret:

            # Display the resulting frame
            # cv2.imshow('Frame', frame)
            cv2.imwrite('data/frames/' + subject + '_' + vid_label + '_' + str(n) + '.png', frame)
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


def main(argv):
    global opts
    help_text = 'data_cleaning.py -[s:]'
    try:
        opts, args = getopt.getopt(argv, "hs:d")
    except getopt.GetoptError:
        print(help_text)
        sys.exit()
    subject = "*"
    for opt, arg in opts:
        if opt == '-h':
            print(help_text)
            sys.exit()
        elif opt == '-s':
            subject = str(arg)
        elif opt == '-d':
            delete_frames()
            sys.exit()

    get_videos(subject)


if __name__ == "__main__":
    main(sys.argv[1:])
