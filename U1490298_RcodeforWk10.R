# Load the CSV data file provided for the assignment. File is already stored in R working directory.
reuters<-read.csv("reutersCSV")


# Check for any records that are not marked for training or testing - 722 found
lostdata<-subset(reuters,(reuters$purpose!="test"& reuters$purpose!="train"))
# remove them from the data
reuterscln<-subset(reuters,(reuters$purpose!="not-used") )
#Missing data check
missing<-reuterscln[!complete.cases(reuterscln),]

#Extract topic labels from the binary coding and implment strategy to duplicate rows with more than 
# on topic assignment .
#looping code to convert the topic matrix into vector assignment of topics #to each row of reuters data.
#topics4 is a data frame of the topics where there 1's in the binary        #representation of the 10 most commmon topics

topics4<- c("pid","fileName","purpose","doc.title","doc.text","doc.topic")

for (i in  1:nrow(reuterscln)){  
  
  for(j in 4:13){
    
    if (reuterscln[i,j]==1) {
      temp_topic <- colnames(reuterscln)[j]
      topics4<- rbind(topics4,cbind(reuterscln[i,c(1,2,3,14,15)],temp_topic))            
      
    } # end of main if condition
    
  }# end of j loop 
  
}  #end of i loop

# Merge the title and main text fields into a new column in topics4 object
topics4$alltxt<-paste(topics4$doc.title, topics4$doc.text)

#Remove the now redundant title and doc.text columns and rename topic Column.

topics4<-topics4[,-(4:5),drop=FALSE]
topics4<-rename(topics4, c("temp_topic"="topic"))

#Create objects comprising of just the text and just the topics that will be used for feature engineering and 
#classification tasks

justtxts<-topics4[,5,drop=FALSE]
justtopics<-topics4[,4,drop=FALSE]

# Create a customised stopwords object to improve removal of redundant information such as month info
myStopwords <- c(stopwords('english'), "available", "via","Reuter","reuter","report","reported","january",
"february","march","april","may","june","july","september","october","november","december",
"January","February","March","April","May","June","July","September","October","November","December","marchapril")

# Create a main document corpus for transformations and analysis 
news_corpus <- Corpus(VectorSource(justtxt))

corpus_clean <- tm_map(news_corpus, removeNumbers)
#Remove all numbers from the text
corpus_clean <- tm_map(corpus_clean, removeWords, myStopwords)
#Remove common non information carrying words in English
 corpus_clean <- tm_map(corpus_clean, removePunctuation)
#Remove punctuation from the sentences 
corpus_clean<-tm_map(corpus_clean, stemDocument)
# Stem words
corpus_clean <- tm_map(corpus_clean, tolower)
# convert all words to lower case
 corpus_clean <- tm_map(corpus_clean, stripWhitespace)
#Remove excess spaces between words and characters 

# For part of speech tagging and named entity recognition processes - please see main report
# these processes would not run successfully on an i7 laptop and were run on a small corpus to 
#demonstrate knowledge of how they are supposed to work

#Named entity recognition - was not used on the full corpus as it was too large. A sample was prepared on 
#a smaller corpus - see report for details
entities <- function(doc, kind) {
  +     s <- doc$content
  +     a <- annotations(doc)[[1]]
  +     if(hasArg(kind)) {
    +         k <- sapply(a$features, `[[`, "kind")
    +         s[a[k == kind]]
    +     } else {
      +         s[a[a$type == "entity"]]
      +     }
  + }


> corpus_annotations <- annotate(news_corpus, list(sent_ann, word_ann,person_ann,location_ann,organization_ann))

> corp_doc <- AnnotatedPlainTextDocument(news_corpus ,corpus_annotations)
> entities(corp_doc,kind="person")


#*****Part of speech tagging***********************

#Need sentence and word token annotations.
sent_token_annotator <- Maxent_Sent_Token_Annotator()
word_token_annotator <- Maxent_Word_Token_Annotator()
wordsann <- annotate(corpus_clean, list(sent_token_annotator, word_token_annotator))

pos_tag_annotator <- Maxent_POS_Tag_Annotator()
pos_tag_annotator
posReut <- annotate(corpus_clean, pos_tag_annotator, wordsann)

## Determine the distribution of POS tags for word tokens.
POSwords <- subset(posReut, type == "word")
tags <- sapply(POSwords$features, `[[`, "POS")
#combine the tags with the corpus words 
POSReuters<-paste(corpus_clean,tags)


#***************CREATION of DOCUMENT TERM MATRICES**************************

Mycorpus <- tm_map(corpus_clean, PlainTextDocument)

dtm<- DocumentTermMatrix(Mycorpus, control = list(stemming = TRUE, stopwords = TRUE, wordLengths = c(3,Inf),removeNumbers = TRUE, removePunctuation = TRUE,stripWhitespace=TRUE))

#Size of term document matrix is ;
#dim(dtm)
#[1] 26450  9990 #i.e. 26450 words x 9990 documents.

#******************FEATURE ENGINEERING*************************************

# Bag of Words approach - calculate the tf-idf factor
tfidf<-tapply(dtm$v/row_sums(dtm)[dtm$i], dtm$j, mean) *log2(nDocs(dtm)/col_sums(dtm))

mydtm <- dtm[, tfidf >= 0.1]
# dim(mydtm)  to show size of feature set after appling tf-idf factor 
#[1]  9990 17663  judged too high 

# LDA topic models - create 10 topics to reflect the same topics selected for the exercise
ReutLDA<-LDA(dtm,10, method = "Gibbs")
#Represent each topic by the top 50 words in each topic
Terms <- as.vector(terms(ReutLDA, 50))

#Create an LDA feature based doc term matrix using the Terms as a dictionary to restrict the words used 
LDAdtm<- DocumentTermMatrix(Mycorpus,control = list(wordLengths = c(3,Inf),dictionary= Terms))
#> dim(LDAdtm)
#[1] 9990  459

# Create a feature set usint the RWeka interface
BigramTokenizer <- function(x) RWeka::NGramTokenizer(x, RWeka::Weka_control(min = 1, max = 2))

dtm_bigrams <- DocumentTermMatrix(Mycorpus, c( weighting = weightBin,tokenize = BigramTokenizer))
#a very large feature set of bigrams is created which needs reducing
#using a revised tfidf factorr
#> dim(dtm_bigrams)
#[1]   9990 322891

bigramtfidf<-tapply(dtm_bigrams$v/row_sums(dtm_bigrams)[dtm_bigrams$i], dtm_bigrams$j, mean) *log2(nDocs(dtm_bigrams)/col_sums(dtm_bigrams))
#reduces bigram dtm.
Bi_dtm <- dtm_bigrams[, tfidf >= 1.8]
#> dim(Bi_dtm)
#[1] 9990  318 features 

# We now create data for evaluating classifers using just the test data from the three
#feature sets doc term matrices.


LDAclassifiers<- as.data.frame(as.matrix(LDAdtm)) 
LDAclassifiers<-cbind(LDAclassifiers,Cleantxts$purpose)
LDAclassifiers<-cbind(LDAclassifiers,justtopics$topic)
LDAclassifiers<-rename(LDAclassifiers,c("justtopics$topic"="topic","Cleantxts$purpose"="purpose"))
TrainLDAclassifiers<-subset(LDAclassifiers,LDAclassifiers$purpose !="test")

BoWclassifiers<- as.data.frame(as.matrix(mydtm)) 
BoWclassifiers<-cbind(BoWclassifiers,Cleantxts$purpose)
BoWclassifiers<-cbind(BoWclassifiers,justtopics$topic)
BoWclassifiers<-rename(BoWclassifiers,c(“Cleantxts$purpose"="purpose","justtopics$topic"="topic"))
TrainBoWclassifiers<-subset(BoWclassifiers, BoWclassifiers$purpose !="test")

For the Bigrams 
Bigclassifiers<- as.data.frame(as.matrix(Bi_dtm)) 
Bigclassifiers<-cbind(Bigclassifiers,Cleantxts$purpose)
Bigclassifiers<-cbind(Bigclassifiers,justtopics$topic)
Bigclassifiers<-rename(Bigclassifiers,c("justtopics$topic"="topic","Cleantxts$purpose"="purpose"))
TrainBigclassifiers<-subset(Bigclassifiers, Bigclassifiers$purpose !="test")



# 10 fold classifier evaluation code - extracted from week 7 exercises.

predNB<-vector()
true<-vector()
predSV<-vector()
predRF<-vector()

#**************************************************
#For this section the same code is used for running the classifiers on each of the feature sets 
the feature sets are assigned to the variable df and the code was run several times manually
other features sets used were TrainLDAclassifiers and TrainBigclassifiers

df<-TrainBoWclassifiers

#************************************************

#Randomly shuffle the data
dff<-df[sample(nrow(df)),]
#dff<-dff[-1] # remove the first column of file names, it messes up Random Forests function

#Create 10 equally size folds
folds <- cut(seq(1,nrow(dff)),breaks=10,labels=FALSE)

#Perform 10 fold cross validation
for(i in 1:10){
  #Segement  data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- dff[testIndexes, ]
  trainData <- dff[-testIndexes, ]
  
  # RUN the three classifiers to create predictions from their models
  
  testcol<-as.data.frame(testData[,ncol(testData)])# the TRUE classes of the test data
  testcol<-factor(testcol[[1]]) #true class data from test set
  
  
  NB<-naiveBayes(topic~., trainData)# create naive bayes model from training data
  NBP<-predict(NB,testData) # predicted Naive Bayes classes for test data
  #NBP<-factor(NBP[[1]])  # predicted classes from Naive Bayes
  
  
  SVM<-svm(topic~.,data= trainData)
  #print(nlevels(predict(SVM,newdata=testData)))
  #SVP<-as.data.frame(predict(SVM,newdata=testData))
  SVP<-predict(SVM,testData)
  #SVP<-factor(SVP[[1]])   # predicted classes from Support Vectors
  #print(nlevels(SVP))
  
  
  RF<-randomForest(topic~.,data= trainData)
  RFP <- predict(RF,newdata=testData)
 # RFP<-factor(RFP[[1]])    # predicted classes from Random Forests
  
  # add some logic to resample and retest if the prediction levels are not 
  #the same as the test column(true)levels i.e. 8
  
  
  # section creates two aggregated factors for the predicted and true classes  in order to
  #feed them into the confusion matrix functions
 
 true <-(append(true,testcol,after = length(true)))
 true<- factor(true,levels=1:nlevels(testcol),labels = levels(testcol)) # adds the level nameas back
  
  predNB<-(append(predNB,NBP,after = length(predNB)))
  predNB<- factor(predNB,levels=1:nlevels(NBP),labels = levels(NBP)) # adds the level nameas back
 levels(predNB)<- levels(actual)
 predNB[]<- lapply(predNB,as.character)
  
  predSV<-(append(predSV,SVP,after = length(predSV)))
  predSV<- factor(predSV,levels=1:nlevels(SVP),labels = levels(SVP)) # adds the level nameas back
 levels(predSV)<- levels(actual)
 predSV[]<- lapply(predSV,as.character)
  
  predRF<-(append(predRF,RFP,after = length(predRF)))
  predRF<- factor(predRF,levels=1:nlevels(RFP),labels = levels(RFP)) # adds the level nameas back
 levels(predRF)<- levels(actual)
 predRF[]<- lapply(predRF,as.character)
  
  print("NB")
  print(nlevels(predNB))
  print("SVM")
  print(nlevels(predSV))
  print("RF")
  print(nlevels(predRF))
  #print(i)
  #output confusion matrices
  #confusionMatrix(predNB,true)$table
  #confusionMatrix(predRF,true)$table
  
} # end of main for loop to sample and run predicitons  10 times 

CMRF<-confusionMatrix(predRF,true)
CMCls<-CMRF$byClass # isolates the statistics for the RF confusion matrix - specificity and sensitivity
margin.table(CMCls,margin=1) 
CMSV<-confusionMatrix(predSV,true)
CMSVCls<-CMSV$byClass # isolates the statistics for the SV confusion matrix - specificity and sensitivity
margin.table(CMSVCls,margin=1)
CMNB<-confusionMatrix(predNB,true)
CMNBCls<-CMNB$byClass # isolates the statistics for the NB confusion matrix - specificity and sensitivity
margin.table(CMNBCls,margin=1)


#F_RF<-CMCls[,c("Sensitivity","Specificity")]
#F_col<-(CMCls[,2]*2*CMCls[,1])/(CMCls[,2]+CMCls[,1]) # calc of F measure for RF 
#ResultRF<- cbind(F_RF,F_col)

F_NB<-CMNBCls[,c("Sensitivity","Specificity")]
F_colNB<-(CMNBCls[,2]*2*CMNBCls[,1])/(CMNBCls[,2]+CMNBCls[,1]) # calc of F measure for RF 
ResultNB<- cbind(F_NB,F_colNB)

F_SV<-CMSVCls[,c("Sensitivity","Specificity")]
F_colSV<-(CMSVCls[,2]*2*CMSVCls[,1])/(CMSVCls[,2]+CMSVCls[,1]) # calc of F measure for RF 
ResultSV<- cbind(F_SV,F_colSV)

confNB<-confusionMatrix(predNB,true)$table
confSV<-confusionMatrix(predSV,true)$table
confRF<-confusionMatrix(predRF,true)$table

centNB<-diag(confNB)
centSV<-diag(confSV)
centRF<-diag(confRF)

confrowsNB<-margin.table(confNB,margin=1)# expression calculates sums of rows in NB  table 
confcolsNB<-margin.table(confNB,margin=2)# calculates sums of columns in NB table
confrowsSV<-margin.table(confSV,margin=1)# expression calculates sums of rows in NB  table 
confcolsSV<-margin.table(confSV,margin=2)# calculates sums of columns in NB table
confrowsRF<-margin.table(confRF,margin=1)# expression calculates sums of rows in NB  table 
confcolsRF<-margin.table(confRF,margin=2)# calculates sums of columns in NB table


microRFRec<-centRF/confrowsRF # calcs micro ave recall of the RF classifier
microRFPrec<-centRF/confcolsRF  # calcs micro ave precision of the RF classifier
microRF_F<- ((2*microRFRec*microRFPrec))/(microRFRec + microRFPrec) #F measure

microNBRec<-centNB/confrowsNB # calcs micro ave recall of the NB classifier
microNBPrec<-centNB/confcolsNB  # calcs micro ave precision of the NB classifier
microNB_F<- ((2*microNBRec*microNBPrec))/(microNBRec + microNBPrec) #F measure

microSVRec<-centSV/confrowsSV # calcs micro ave recall of the SV classifier
microSVPrec<-centSV/confcolsSV  # calcs micro ave precision of the SV classifier
microSV_F<- ((2*microSVRec*microSVPrec))/(microSVRec + microSVPrec) #F measure

cbind(microNBRec,microNBPrec,microNB_F)
cbind(microSVRec,microSVPrec,microSV_F)
cbind(microRFRec,microRFPrec,microRF_F)


## SVMClassifier evaluation using full training and test  and LDA topic feature sets not cross folds

LDAcontainer<-create_container(LDAdtm,topic,trainSize=1:7199,testSize=7200:9990,virgin=TRUE)
 svm_LDA<-train_model(LDAcontainer, algorithm="SVM",method = "C-classification",kernel= "radial",cost=100,cross=0)
 SVMLDApred<-classify_model(LDAcontainer,svm_LDA)
 LDApredicted<- SVMLDApred[,1]
 levels(LDApredicted)<- levels(actual)
 LDApredicted[]<- lapply(LDApredicted,as.character)
 LDAcm<-confusionMatrix(data=LDApredicted,reference=actual)

#SVMClassifier evaluation using full training and test  and main Bag of Words  feature sets not cross folds
container<-create_container(mydtm,topic,trainSize=1:7199,testSize=7200:9990,virgin=TRUE)
svmfulldata<-train_model(container, algorithm="SVM",method = "C-classification",kernel= "radial",cost=100,cross=0)
SVMpred<-classify_model(container,svmfulldata)
SVMpredfull<-SVMpred[,1]
levels(SVMpredfull)<- levels(actual)
SVMpredfull[]<- lapply(SVMpredfull,as.character)
fullcm<-confusionMatrix(data=SVMpredfull,reference=actual)



#************Clustering************************
#DBscan
LDAclust<-dbscan(LDAdtm,0.4,MinPts=5,method="hybrid")
plot(LDAclust$cluster)

#Hierarchical cluster
LDAdist<-dist(LDAdtm, method = "euclidean", diag = FALSE, upper = FALSE, p = 2)
LDAhclust<-hclust(LDAdist,method="ward.D",members=NULL)
plot(LDAhclust,labels=FALSE)


#Lloyds K means, specify 10 clusters
KM<-kmeans(LDAdtm,10)
hist(KM$cluster)

Kclust<-kmeans(LDAdtm,10,method="centers,algorithm="Lloyd"")
Kclust$betweenss
Kclust$withinss
plot(Kclust$cluster)
plot(Kclust$size)
hist(Kclust$cluster)

#write csv files out for processing in WEKA
LDAmatrix<-as.data.frame(as.matrix(LDAdtm))
 write.csv(LDAmatrix,file="LDAmatrix.csv")
write.csv(justtopics,file="justtopics.csv")
