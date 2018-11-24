import glob
import os

filelist=glob.glob("../data/chi-selfie/*/*.mp4")

mapping=['a','d','f','h','n','sa','su']
for f in filelist:
    # print(f[-8:-7])
    if f[-8:-7]=='0':
        # print(f)
        os.rename(f,f[:-8]+mapping[(int(f[-8:-4])-1)%7]+str(int((int(f[-8:-4])-1)/7)+1)+'.mp4')
    
