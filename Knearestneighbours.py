import nltk
import docx
import os
import string
import math
import re

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


spath = r"G:\project\All Cancer  Files"

voca=[]
text=''
temp_text=''
ct=0

for i in os.listdir(spath) :
    if i.endswith('.docx'):
        if i[:2] !='~$':
            doc = docx.Document(spath+'\\'+i)
            for para in doc.paragraphs:
                text = text+' '+para.text
                temp_text = temp_text+' '+para.text
            voca.append(temp_text)
            temp_text=''

#Classify Y

def find_y(voca):
    c=len(voca)
    y=[1 for i in range(0,c)]
    for i in range(0,c) :
        Word=nltk.word_tokenize(voca[i])
        if 'II' in Word:
            y[i]=2
        if 'III' in Word:
            y[i]=3
        if 'IV' in Word:
            y[i]=4
        if 'V' in Word:
            y[i]=5
        if 'VI' in Word:
            y[i]=6    
        
    return y
    

# text is your corpus

def clean_text(text):
    w_text = nltk.word_tokenize(text)
    
    punc_text=[]
    for i in w_text:
        if i not in string.punctuation:
            punc_text.append(i) 

    for i in punc_text:
        i=i.lower()
    
    stop_words=set(stopwords.words('english'))

    non_stop_text=[]        
    for i in punc_text:
        if i not in stop_words:
            non_stop_text.append(i)

    ps=PorterStemmer()

    stem_text=[]
    for i in non_stop_text:
        stem_text.append(ps.stem(i))
    
    unique_text = list(set(stem_text))

    return unique_text

def doc_words_cleaning(voca):       # cleans and preprocesses each document 
    doc_words = []    
    for i in range(len(voca)):
        doc_words.append(clean_text(voca[i]))
    return doc_words

def bag_of_words(doc_words,doc_temp,vocabulary):    # boolean representation 
    bool_vec = []
    temp_vec=[]
      
    for i in range(len(doc_words)):
        for j in range(len(vocabulary)):
            if vocabulary[j] in doc_words[i]:
               temp_vec.append(1)
            else:
                temp_vec.append(0)      
        
        bool_vec.append(temp_vec)       
        temp_vec=[]  
        
    return bool_vec

def doc_term_freq(boo):
    doc_freq=[]
    for i in range(len(boo)):
        summ=0
        for j in range(len(boo[0])):
            if boo[i][j] == 1:
                summ=summ+1
        doc_freq.append(summ)
    return doc_freq

# count of no of vocabulary words in each document

def count_voca_words(bove):
    a=[]
    for i in range(len(bove)):
        a.append(bove[i].count(1))    
    return a

# term frequency [NAIVE] :-  
    
    #This gives the frequency of every word in the

def term_frequency_naive(doc_words,vocabulary):   
    tf=[]
    temp=[]
    for i in range(len(doc_words)):
        for j in vocabulary:
            temp.append(doc_words[i].count(j))
        tf.append(list(zip(temp,vocabulary)))
        temp=[]
    
    return tf            

# tf - idf term frequency 

def tf(doc_words,vocabulary):
    tf=[]
    temp=[]
    for i in range(len(doc_words)):
        for j in vocabulary:
            temp.append((doc_words[i].count(j))/len(doc_words[i]))
        tf.append(temp)
        temp=[]
    return tf

def idf(doc_words,vocabulary):
    doc_count = 0
    idf=[]
    for i in vocabulary:
        for j in range(len(doc_words)):
            if i in doc_words[j]:
                doc_count = doc_count + 1
        idf.append(math.log(len(doc_words)/(doc_count)))
        doc_count = 0 
    
    return idf

def tf_idf_score_calc(tf,idf):
    tf_idf = []
    temp=[]
    for i in range(len(tf)):
        for j in range(len(idf)):
            if tf[i][j] !=0 and idf[j] !=0:
                temp.append(tf[i][j]*idf[j])
            else:
                temp.append(0)
        tf_idf.append(temp)
        temp=[]
    
    return tf_idf


def Roc_curve(y_test,y_score,n_classes):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc
    from scipy import interp

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    l=len(set(y_score))

    for i in range(l):
        fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(y_score))[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(l)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(l):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr = mean_tpr/(n_classes-1)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw=2
    plt.figure(figsize=(8,5))
    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='green', linestyle=':', linewidth=4)

    colors = cycle(['chocolate', 'aqua', 'darkorange', 'cornflowerblue',   'cadetblue','burntsienna','cornflowerblue'])
    for i, color in zip(range(l), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('Random Guess',(.5,.48),color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for K nearest neighbours')
    plt.legend(loc="lower right")
    plt.show()

   


   
def knearestneigbor(x,y):
    #splitting Training Data and Test Data
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state = 0)


    #Knn Neighbours Prediction
    from sklearn.neighbors import KNeighborsClassifier
    model=KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train,y_train)
    result_knn=model.predict(x_test)

    #Model report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report

    results = confusion_matrix(y_test, result_knn)
    print('Confusion Matrix :')
    print(results)

    print('Accuracy Score :',accuracy_score(y_test,result_knn))

    print('Report :' )
    print (classification_report(y_test,result_knn))
    Roc_curve(y_test,result_knn,len(list(set(y))))
    #Roc_curve(results,len(list(set(y))))

vocabulary = clean_text(text)

doc_words = doc_words_cleaning(voca)

bow = bag_of_words(doc_words,voca,vocabulary)

y=find_y(voca)

doc_freq = count_voca_words(bow)   

term_freq_naive = term_frequency_naive(doc_words,vocabulary)

term_frequency = tf(doc_words,vocabulary)

inverse_doc_freq = idf(doc_words,vocabulary)

tf_idf_score = tf_idf_score_calc(term_frequency,inverse_doc_freq)

print("\nBag of Words:")
r=knearestneigbor(bow,y)


print("\nTerm Frequency:")
knearestneigbor(term_frequency,y)

print("\ntf_idf:")
knearestneigbor(tf_idf_score,y)
