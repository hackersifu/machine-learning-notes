import os
import tempfile

def disPickle(fname):
    contents = []
    f = open(fname,'rb')
    while f.tell() != os.fstat(f.fileno()).st_size:
        temp = tempfile.TemporaryFile("w+")
        try:
            pickletools.dis(f, temp)
        except Exception as e:
            print(e)
            break
        temp.seek(0)
        contents.append(temp.read())
        temp.close()
    return contents
    

# just a pickle
print('\n0')
pickles = disPickle('models/model0.pkl')
print('num pickles:',len(pickles))

# pytorch old format, end is some non pickle data
print('\n1')
pickles =  disPickle('models/model1.pt')
print('num pickles:',len(pickles))
# print(pickles[2])

# pytorch old format, end is some non pickle data
print('\n2')
pickles =  disPickle('models/model2.pt')
print('num pickles:',len(pickles))
# print(pickles[0])

# pytorch new format
print('\n3')
!file models/model3.pt
# file shows us it's a zip file so we open that and pull out the pkl
import zipfile
zf = zipfile.ZipFile('models/model3.pt')
print(zf.namelist())
content = zf.open('model3-statedict/data.pkl','r').read()
# write that pkl to a file we can disassemble with our function
open('test.pkl','wb').write(content)
pickles =  disPickle('test.pkl')
print('num pickles:',len(pickles))

# numpy is harder
print('\n4')
!file models/model4.npy
# file shows it's a numpy file if the extension is changed
!head models/model4.npy
# head shows that there is a dict followed by a new line we can strip
content = open('models/model4.npy','rb').read()
content = content.partition(b'\n')[2]
open('test2.pkl','wb').write(content)
# we strip the header and write the remaining contenst to a temporary
# file we can disasemble withour existing funciton
pickles =  disPickle('test2.pkl')
print('num pickles:',len(pickles))

# just a pickle created by joblib
print('\n5')
pickles =  disPickle('models/model5.joblib')
print('num pickles:',len(pickles))