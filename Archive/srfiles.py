import os

for files in os.listdir('/user/skansal/home/Results2/'):
    result = []
    for lines in open(os.path.join('/user/skansal/home/Results2', files)):
        lines = lines.strip()
        result.append(lines)

    result = list(set(result))
    outputfilename = os.path.join('/user/rpandey/home/shaira', files)
    with open(outputfilename, 'w') as outfile:
        for item in result:
            outfile.write("%s\n" % item)
    print ("hurrah %s " % files)
    result = []
