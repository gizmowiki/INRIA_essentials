import os

total_frames = 0
for lines in open('/home/rpandey/dataset/dataset_jan_26/files_chalearn.txt'):
    lines = lines.strip()
    # print lines
    query = "ffmpeg -i "+lines+" -vcodec copy -acodec copy -f null /dev/null 2>&1 | grep 'frame=' | cut -f 2 -d ' '"
    result = os.popen(query).read()
    result = result.strip()
    print ("Video files path: %s Total frames: %s " % (lines, result))
    if result.isdigit():
        total_frames += int(result)

print ("Total frames in dataset: %s " % total_frames)
