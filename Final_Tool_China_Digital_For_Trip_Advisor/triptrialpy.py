from flask import Flask, request, render_template
import requests
import os
import re
import pandas as pd
from requests_html import HTMLSession
from bs4 import BeautifulSoup
from googletrans import Translator
session = HTMLSession()
trans = Translator()
from textblob import TextBlob
import itertools
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords,wordnet
from nltk.stem.porter import PorterStemmer
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.tree import DecisionTreeClassifier
#----------------------------------------------------------------------------------------------------------------------------------------------------#

app = Flask(__name__)
#----------------------------------------------------------------------------------------------------------------------------------------------------#

@app.route('/')
def my_form():
    return render_template('trial.html')
#----------------------------------------------------------------------------------------------------------------------------------------------------#

def keywordSearch(text):
    #keyword = input("输入中文关键字: ")
    keyword = trans.translate(text)
    keyword = keyword.text
    finalkeyword = ""
    if " " in keyword:
        keywords = keyword.split(" ")
        for i in range(len(keywords)):
            finalkeyword = finalkeyword + keywords[i] + '%20'
        finalkeyword = finalkeyword[:-3]
    else:
        finalkeyword = keyword
    spliturl = "https://www.tripadvisor.cn/Search?q=&searchSessionId=C77B00A2C532503BA05FB587D9CBA5991594309950920ssid&sid=0550FE94C20D5CC7ED543C49EB038B3F1594319147181&blockRedirect=true&ssrc=a&geo=1&rf=1"
    part1url = spliturl[:36]
    part2url = spliturl[36:]
    refurl = part1url + finalkeyword + part2url
    return(refurl,keyword)
#----------------------------------------------------------------------------------------------------------------------------------------------------#
def headerrequests(refurl):

    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36'
    headers = { 'User-Agent' : user_agent, 
           "Referer": refurl,
           'content-type': 'Application/json',
          'X-Requested-With': 'XMLHttpRequest'}
    return headers

#----------------------------------------------------------------------------------------------------------------------------------------------------#
def categoryURLGenerator(refurl, categoryCode):
    a = refurl.split("&ssrc=")
    rem = "&ssrc="+categoryCode+a[1][1:]
    return a[0]+rem

#----------------------------------------------------------------------------------------------------------------------------------------------------#
def keywordchecker(refurl):
    headers = headerrequests(refurl)
    x = requests.get(refurl, headers=headers)
    soup = BeautifulSoup(x.text, 'html.parser')
    print(soup)
    no_result =soup.findAll(class_="no-results-content")
    if len(no_result)!=0:
        print("Hello! No Results found, Please enter proper keyword!")
        newrefurl,keyword = keywordSearch()
        x = keywordchecker(newrefurl)
    elif soup.findAll(class_="review_count")==None:
        print("*****")
        categoryCode=['h','e','A','g','al','NA']
        for i in categoryCode:
            if i!='NA':
                newCategoryUrl= categoryURLGenerator(refurl, i)
                x = requests.get(newCategoryUrl, headers=headers)
                soup = BeautifulSoup(x.text, 'html.parser')
                if soup.findAll(class_="review_count")!=None:
                    x= soup.find_all('a', class_='review_count')
                    break
            else:
                print("No Results found, Please enter proper keyword!")
                newrefurl,keyword = keywordSearch()
                x = keywordchecker(newrefurl)
    else:
        x = soup.find_all('a', class_='review_count')
    return x
#----------------------------------------------------------------------------------------------------------------------------------------------------#
def pagefunction(baseurl, pagenum):
    a = baseurl.split("Reviews")
    Pagenumber = "Reviews-or"+str(pagenum)
    key = Pagenumber+a[1]
    #print(a[0] + key)
    return a[0]+key
#----------------------------------------------------------------------------------------------------------------------------------------------------#
def makefinalurl(y):
	try:
		finalurl = "https://www.tripadvisor.cn" + y[0]['href']
	except:
		finalurl = "No Results Found"
	return finalurl
#----------------------------------------------------------------------------------------------------------------------------------------------------#
def getreviews(urls):
    date_lst=[]
    urls1 = urls
    reviews_lst=[]
    searchurllist=[]
    urls = requests.get(urls)
    soup = BeautifulSoup(urls.text, 'html.parser')
    chinese_review = []
    for tag in soup.findAll(class_="ui_column is-9"):
        if '通过移动设备发表' in tag.text:
            content = tag.text.replace('\n', '').split('的点评 通过移动设备发表')
        else:
            content = tag.text.replace('\n', '').split('的点评')
        cdate = trans.translate(content[0])
        creview = trans.translate(content[1])
        chinese_review.append(content[1])
        date_lst.append(cdate.text)
        reviews_lst.append(creview.text)
        searchurllist.append(urls1)
    return date_lst,reviews_lst,chinese_review,searchurllist

#----------------------------------------------------------------------------------------------------------------------------------------------------#
@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    refurl,keyword = keywordSearch(text)
    x = keywordchecker(refurl)
    finalurl = makefinalurl(x)
    if(finalurl!="No Results Found"):
    	i=0
    	newgenurl=""
    	urllist=[]
    	urllist.append(finalurl)
    	x=""
    	sp=["Hello"]
    	while(len(sp)!=0):
    		i+=10
    		newgenurl = pagefunction(finalurl,i)
    		#print(newgenurl)
    		urlnew = requests.get(newgenurl)
    		soup = BeautifulSoup(urlnew.text, 'html.parser')
    		sp=soup.findAll(class_="ui_column is-9")
    		urllist.append(newgenurl)
    	del urllist[-1]
    	finalreviewslist=[]
    	finaldatelist=[]
    	finalchineselist=[]
    	finalsearchurllist=[]
    	for j in range(len(urllist)):
    		returnval = getreviews(urllist[j])
    		finalreviewslist.append(returnval[1])
    		finaldatelist.append(returnval[0])
    		finalchineselist.append(returnval[2])
    		finalsearchurllist.append(returnval[3])
    	sentimentscore = []
    	for m in range(len(finalreviewslist)):
    		temp=[]
    		for n in range(len(finalreviewslist[m])):
    			temp.append(TextBlob(str(finalreviewslist[m][n])).sentiment[0])
    		sentimentscore.append(temp)
    	sentimentval=[]
    	for x in range(len(sentimentscore)):
    		tempval=[]
    		for y in range(len(sentimentscore[x])):
    			if sentimentscore[x][y]>0.2:
    				tempval.append("Positive")
    			elif sentimentscore[x][y]<-0.2:
    				tempval.append("Negative")
    			else:
    				tempval.append("Neutral")
    		sentimentval.append(tempval)
    	datelist = list(itertools.chain.from_iterable(finaldatelist))
    	reviewlist = list(itertools.chain.from_iterable(finalreviewslist))
    	sentimentlist = list(itertools.chain.from_iterable(sentimentval))
    	chineselist = list(itertools.chain.from_iterable(finalchineselist))
    	searchurllistlist = list(itertools.chain.from_iterable(finalsearchurllist))
    	final_df = pd.DataFrame(
	    	{'Date': datelist,
	    	'Search Url':searchurllistlist,
	     	'English Reviews': reviewlist,
	     	'Chinese Reviews' : chineselist,
	     	'Sentiment Using TextBlob Scores': sentimentlist
	     	#'Sentiment Using ML Model' : y_pred
	     	})
    	df_train1 = pd.read_excel("reviews_for_train.xlsx")
    	df_train2 = pd.read_excel("200717 Additional Example Reviews Practera.xlsx")
    	df_train2.rename(columns = {'Unnamed: 2':'label'}, inplace = True)
    	df_train2.rename(columns = {'Translation to English':'Reviews'}, inplace = True)
    	df_train2 = df_train2.drop(['Review in Chinese'], axis=1)
    	frames = [df_train1, df_train2]
    	df_train = pd.concat(frames)
    	df_train['label'] = df_train['label'].replace([' Negative','−','N'], 'Negative')
    	df_train['label'] = df_train['label'].replace(['NeutralNegativenegative','n','neutral'], 'Neutral')
    	df_train['Reviews'] = [word.lower() for word in df_train['Reviews']]
    	final_df['English Reviews_New'] = [word.lower() for word in final_df['English Reviews']]
    	df_train['Reviews'] = df_train['Reviews'].apply(lambda x: re.sub(r'\d+','',x))
    	final_df['English Reviews_New']  = final_df['English Reviews_New'].apply(lambda x: re.sub(r'\d+','',x))
    	df_train['Reviews'] = df_train['Reviews'].str.replace('[^\w\s]','')
    	final_df['English Reviews_New'] = final_df['English Reviews_New'].str.replace('[^\w\s]','')
    	df_train['Reviews'] = df_train['Reviews'].apply(lambda x: x.strip())
    	final_df['English Reviews_New'] = final_df['English Reviews_New'].apply(lambda x: x.strip())
    	df_train['Reviews'] = df_train['Reviews'].apply(lambda x: re.sub(r'\s+',' ',x))
    	final_df['English Reviews_New']  = final_df['English Reviews_New'] .apply(lambda x: re.sub(r'\s+',' ',x))
    	df_train['Reviews'] = [word_tokenize(word) for word in df_train['Reviews']]
    	final_df['English Reviews_New']  = [word_tokenize(word) for word in final_df['English Reviews_New']]
    	stop_words = set(stopwords.words('english'))
    	df_train['Reviews'] = df_train['Reviews'].apply(lambda x: [item for item in x if item not in stop_words])
    	final_df['English Reviews_New'] = final_df['English Reviews_New'].apply(lambda x: [item for item in x if item not in stop_words])
    	porter = PorterStemmer()
    	df_train['Reviews'] = df_train['Reviews'].apply(lambda x: [porter.stem(y) for y in x])
    	final_df['English Reviews_New'] = final_df['English Reviews_New'].apply(lambda x: [porter.stem(y) for y in x])
    	df_train['Reviews']=df_train['Reviews'].apply(lambda x: " ".join(x))
    	final_df['English Reviews_New']=final_df['English Reviews_New'].apply(lambda x: " ".join(x))
    	le = preprocessing.LabelEncoder()
    	df_train['encoded_label'] = le.fit_transform(df_train['label'])
    	x_train = df_train['Reviews']
    	y_train = df_train['encoded_label']
    	x_test = final_df['English Reviews_New']
    	tfidf_vect = TfidfVectorizer(max_features=5000)
    	xtrain_tfidf =  tfidf_vect.fit_transform(x_train)
    	xtest_tfidf =  tfidf_vect.transform(x_test)
    	clf = DecisionTreeClassifier(random_state=0)
    	clf.fit(xtrain_tfidf,y_train)
    	y_pred = clf.predict(xtest_tfidf)
    	y_pred = le.inverse_transform(y_pred) 
    	final_df1 = pd.DataFrame(
	    	{'Date': datelist,
	    	'Search Url':searchurllistlist,
	     	'English Reviews': reviewlist,
	     	'Chinese Reviews' : chineselist,
	     	'Sentiment Using TextBlob Scores': sentimentlist,
	     	'Sentiment Using ML Model' : y_pred
	     	})
    	final_df1['Keyword'] = keyword
    	final_df1["Platform Name"] = "Trip Advisor China"
    	final_df1 = final_df1[['Date','Keyword','Platform Name','Search Url','Chinese Reviews','English Reviews','Sentiment Using TextBlob Scores','Sentiment Using ML Model']]
    	filenameforcsv = "Reviews For "+keyword+" From Trip Advisor China.csv"
    	final_df1.to_csv(filenameforcsv)
    	returntext = "The file has been downloaded in your local machine"
    else:
    	returntext = "No Results Found"
    return returntext

#print(sentimentval)
    #processed_text = text.upper()
#----------------------------------------------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    app.run(debug=True)