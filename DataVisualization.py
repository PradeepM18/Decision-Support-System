import docx
doc=docx.Document('G:\\AMUDHA MAMMO.docx')

i=0
true=1
text=''
while true:
    try:
        text=text+doc.paragraphs[i].text
        i=i+1
	
    except IndexError:
        print("File index exceeded and read successfully....")
        true=0

print(text)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

stop_words=set(stopwords.words('english'))

words=word_tokenize(text)

punctuated_sentence=[]

for i in words:
    if i not in string.punctuation:
        punctuated_sentence.append(i)
        
filtered_sentence=[]

for i in punctuated_sentence:
    if i not in stop_words:
        filtered_sentence.append(i)
        #filtered_sentence+" "+i
print("After removing stop words: ",filtered_sentence)


from nltk.stem import PorterStemmer

ps = PorterStemmer()
stemmed_sentence=[]
for w in filtered_sentence:
    stemmed_sentence.append(ps.stem(w))
print("After stemming: ",stemmed_sentence)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

text=""
for i in stemmed_sentence:
    text=text+" "+i
print("Final text: ",text)

plt.figure(figsize=(20,10))
wordcloud = WordCloud(background_color="white",mode="RGB",width=2000,height=1000).generate(text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

          


    
