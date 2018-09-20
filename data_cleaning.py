import getopt
import glob
import sys

import cv2


def get_videos(subject="*"):
    video_files = glob.glob("data/savee/AudioVisualClip/" + subject + "/*.avi")
    print(video_files)
    for video in videofiles:
        vid2frames(video,)


def vid2frames(path="data/savee/AudioVisualClip/DC/a1.avi", subject="DC"):
    cap = cv2.VideoCapture(path)
    # print(cap)
    n = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        # print(ret)
        if ret:

            # Display the resulting frame
            # cv2.imshow('Frame', frame)
            cv2.imwrite('data/frames/' + subject + '_a1_' + str(n) + '.png', frame)
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
    try:
        opts, args = getopt.getopt(argv, "hs:")
    except getopt.GetoptError:
        print('data_cleaning.py -[s:]')
        sys.exit()
    subject = "*"
    for opt, arg in opts:
        if opt == '-h':
            print('data_cleaning.py -[s:]')
            sys.exit()
        elif opt == '-s':
            subject = str(arg)

    get_videos(subject)


if __name__ == "__main__":
    main(sys.argv[1:])
