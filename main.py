import os
from similarity import similarity
import nltk
import numpy as np
import time

start = time.time()

data_PATH=r'C:\Users\T01144\Desktop\T01144\IDLE Scripts\Text Analysis\dataset'

file_name = []
file_text = []
for filename in os.listdir(data_PATH):
    if filename.endswith(".txt"):
        
        with open(data_PATH+'\\'+filename) as ff:
            file_name.append(filename)
            #print("For", filename)
            
            text = ff.readlines()[0]
            #print(text)
            sent_text = nltk.sent_tokenize(text)
            file_text.append(sent_text)

#print(file_text)
#print("")

simList = []

for i in range(0,len(file_name)):
    for j in range(0, len(file_name)):
        if i!=j:
           
           sentList1=file_text[i]
           sentList2=file_text[j]
           
           print(file_name[i],"&",file_name[j])

           for sentList1Text in sentList1:
               max = 0
               sim=0
               comparedStatement = None
               for sentList2Text in sentList2:
                   sim = similarity(sentList1Text,sentList2Text)
                   if sim>0.5:
                       if max<sim:
                           max = sim
                           comparedStatement = sentList2Text
               print(sentList1Text,"&",comparedStatement)
               print(max)
               simList.append(max)

           simListArr = np.array(simList)
           print("Similarity:",np.sqrt(np.mean(simListArr**2)))
           print("")

# Calculate execution time
end = time.time()
dur = end-start
print("")
if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")
