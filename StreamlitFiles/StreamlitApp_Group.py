import streamlit as st
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
import altair as alt
import string
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import ngrams
from nltk import corpus




st.markdown("<h1 style='text-align: center; color: purple;'>Text Analytics Group Assignment</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: purple;'>Sentiment Analysis for the dataset</h3>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: purple;'>Harsimar Singh Arora - 12120011</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: purple;'>Priya Ranjan Kar - 12120081</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: purple;'>Rohini Purnima - 12120027</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: purple;'>Mohit Kothari - 12120035</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: purple;'>Tipu Thakur - 12120032</h5>", unsafe_allow_html=True)

DataToBeAnalyzed = None
AnalyzedData = None
WordList = None
WordCloudFig = None


#Defining Functions=
def GetRequiredData(dataset_name):
    if dataset_name == "Dataset":
        return ('write',FetchedDataset)
    elif dataset_name == "Valence Distribution":
        return ('plot',FetchValenceDistribution())
    elif dataset_name == "Word Cloud":
        return('plot',FetchWordCloud())
    elif dataset_name == "Bigrams":
        return('write',FetchNgrams(2))
    elif dataset_name == "Bigrams Visualization":
        return('altChart',FetchNgramVisualization(2,10))
    elif dataset_name == "Trigrams":
        return('write',FetchNgrams(3))
    elif dataset_name == "Trigrams Visualization":
        return('altChart',FetchNgramVisualization(3,3))
    elif dataset_name == "Sentiment Distribution":
        return('plot',FetchHistogramForSentiment())
    else:
        return (None,None)
def FetchDataset():
    return pd.read_csv(uploadColumn,encoding = "ISO-8859-1")
def FetchValenceDistribution():
    global DataToBeAnalyzed
    global AnalyzedData
    if DataToBeAnalyzed is None:
        DataToBeAnalyzed = DoPreProcessing()
    if AnalyzedData is None:
        AnalyzedData = DocumentFromCorpusAnalyzer(DataToBeAnalyzed)
        AnalyzedData.reset_index(inplace = True, drop = True)
    x = pd.Series([x for x in range(len(AnalyzedData))])
    y = AnalyzedData.Compound
    plt.figure(figsize=[7,4])
    ValenceFigure,ax = plt.subplots()
    ax.bar(x, y)
    #fig.xlabel("Sentence Number")
    #fig.ylabel("Compound Valence score")
    #fig.title("Valence of Uber Reviews by Sentence.")
    return ValenceFigure
RemoveEmoji = lambda x: re.sub("<[a-zA-Z0-9+]*>","",x).strip()
RemovePunctuations = lambda x: re.sub("[']","",re.sub('[“!"#$%&()*+,-.:;<=>?@[\]^_`{|}”~]+',"",x)).strip()
ConvertLower = lambda x: x.lower()
def DoPreProcessing():
    #inputData = FetchDataset()
    requiredData = FetchedDataset[sentiment_columns]
    requiredData['SentimentData'] = requiredData.apply(lambda x: '\n'.join(x.dropna().astype(str)),axis=1)
    requiredData.SentimentData = requiredData.SentimentData.apply(RemoveEmoji)
    requiredData.SentimentData = requiredData.SentimentData.apply(RemovePunctuations)
    requiredData.SentimentData = requiredData.SentimentData.apply(ConvertLower)
    return requiredData.SentimentData
def DocumentAnalyzer(document):
    sentenceList = sent_tokenize(document)
    documentPolarity = []
    sentenceIndex = []
    for i in range(len(sentenceList)):
        sentencePolarity = analyzer.polarity_scores(sentenceList[i])
        documentPolarity.append(sentencePolarity)
        sentenceIndex.append(i)
    documentDataframe = pd.DataFrame(documentPolarity)
    documentDataframe.insert(0, 'SentenceIndex', sentenceIndex)
    documentDataframe.insert(documentDataframe.shape[1], 'Sentence', sentenceList)
    documentDataframe.rename(columns={'neg':'NegativeScore','pos':'PositiveScore','neu':'NeutralScore','compound':'Compound'}, inplace=True)
    return(documentDataframe)
def DocumentFromCorpusAnalyzer(corpus):
    SentimentData = pd.DataFrame(corpus)
    analyzedDataframe = pd.DataFrame(columns=['DocumentIndex', 'SentenceIndex', 'NegativeScore', 'NeutralScore', 'PositiveScore', 'Compound', 'Sentence'])    
    for i1 in range(len(SentimentData)):
        document = str(SentimentData.iloc[i1].SentimentData)
        analyzedDocument = DocumentAnalyzer(document)
        analyzedDocument.insert(0, 'DocumentIndex', i1)
        analyzedDataframe = pd.concat([analyzedDataframe, analyzedDocument], axis=0)
    return(analyzedDataframe)
def FetchWordList():
    WordList = []
    for sentence in AnalyzedData.Sentence:
        for word in word_tokenize(sentence):
            if re.search('[a-zA-Z0-9]',word):
                WordList.append(word)
    stopWords = corpus.stopwords.words('english')
    WordList = [i for i in WordList if i not in stopWords]
    return WordList
def FetchWordCloud():
    global DataToBeAnalyzed
    global AnalyzedData
    global WordList
    global WordCloudFig
    if DataToBeAnalyzed is None:
        DataToBeAnalyzed = DoPreProcessing()
    if AnalyzedData is None:
        AnalyzedData = DocumentFromCorpusAnalyzer(DataToBeAnalyzed)
        AnalyzedData.reset_index(inplace = True, drop = True)
    if WordList is None:
        WordList = FetchWordList()
    if WordCloudFig is None:
        cloudData = ''
        for word in WordList:
            lowerWord = word.lower().strip()
            cloudData += word + " "
        cloudData = cloudData.replace('uber','').replace('apps','').replace('drivers','').replace('rides','').replace(
            'app','').replace('driver','').replace('ride','')
        wCloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                min_font_size = 10).generate(cloudData)
        plt.figure(figsize=[10,10])
        WordCloudFig,ax = plt.subplots()
        ax.imshow(wCloud)
    return WordCloudFig
def FetchNgrams(n):
    global DataToBeAnalyzed
    global AnalyzedData
    global WordList
    global WordCloudFig
    if DataToBeAnalyzed is None:
        DataToBeAnalyzed = DoPreProcessing()
    if AnalyzedData is None:
        AnalyzedData = DocumentFromCorpusAnalyzer(DataToBeAnalyzed)
        AnalyzedData.reset_index(inplace = True, drop = True)
    if WordList is None:
        WordList = FetchWordList()
    Bigrams = pd.Series(ngrams(WordList,n)).value_counts()
    return Bigrams
def FetchNgramVisualization(n,MinQuantity):
    Bigrams = FetchNgrams(n)
    BigramsMaxUsed = Bigrams[Bigrams > MinQuantity]
    bigramDf = BigramsMaxUsed.to_frame()
    NGramName = 'Bigram' if n==2 else 'Trigram'
    bigramDf = bigramDf.reset_index(level=0).rename(columns={'index':NGramName,0:'Quantity'})
    chart = alt.Chart(bigramDf).mark_bar().encode(x=alt.X(NGramName+':N',sort='-y'),y='Quantity')
    return chart
def FetchHistogramForSentiment():
    global DataToBeAnalyzed
    global AnalyzedData
    if DataToBeAnalyzed is None:
        DataToBeAnalyzed = DoPreProcessing()
    if AnalyzedData is None:
        AnalyzedData = DocumentFromCorpusAnalyzer(DataToBeAnalyzed)
        AnalyzedData.reset_index(inplace = True, drop = True)
    y = AnalyzedData.Compound
    plt.figure(figsize=[7,4])
    Histogram,ax = plt.subplots()
    ax.hist(y,bins=25)
    return Histogram
#Defining Functions Finished

sentiment_columns = None
file_details = None
analyzer = None
uploadColumn = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
dataset_name = None
if uploadColumn is not None:
    LoadImportantLibraries()
    file_details = {"filename":uploadColumn.name, "filetype":uploadColumn.type,
                            "filesize":uploadColumn.size}
    st.write(file_details)
    FetchedDataset = FetchDataset()
    AllColumns = list(FetchedDataset.columns)
    AllColumns.append('All')
    sentiment_columns = st.sidebar.multiselect("Select columns to be used for Sentiment Analysis - ",AllColumns)
    if 'All' in sentiment_columns:
        sentiment_columns = FetchedDataset.columns
    if len(sentiment_columns) != 0:
        dataset_name = st.sidebar.selectbox("Select Functionality", ("None", "Dataset", "Valence Distribution", "Word Cloud", "Bigrams", "Bigrams Visualization", "Trigrams", "Trigrams Visualization", "Sentiment Distribution"))
    else:
        dataset_name = st.sidebar.selectbox("Select Functionality", ("None", "Dataset"))


x,y = GetRequiredData(dataset_name)
if x is not None:
    if x == 'write':
        st.write(y)
    elif x == 'altChart':
        st.altair_chart(y, use_container_width=True)
    else:
        st.pyplot(y)