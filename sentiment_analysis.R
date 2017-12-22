library(tm)
library(RSentiment)
library(data.table)
library(wordcloud)
library(ggplot2)

aut_train = fread("train.csv")

eap <- aut_train[author=="EAP"] # Edgar Allan Poe
hpl <- aut_train[author=="HPL"] # HP Lovecraft
mws <- aut_train[author=="MWS"] # Mary Wollstonecraft Shelley

#Cleaning of Data
corpus = Corpus(VectorSource(list(eap$text)))# Get a vector of all the sentences as elements 
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers) 
corpus = tm_map(corpus, stripWhitespace)
corpus = tm_map(corpus, removeWords, stopwords('english'))

#Creating a DTM  and sentiments for Edgar Allan Poe
dtm_eap = DocumentTermMatrix(VCorpus(VectorSource(corpus[[1]]$content)))
freq_eap <- colSums(as.matrix(dtm_eap))

sentiments_eap = calculate_sentiment(names(freq_eap))
sentiments_eap = cbind(sentiments_eap, as.data.frame(freq_eap))

sent_pos_eap = sentiments_eap[sentiments_eap$sentiment == 'Positive',]
sent_neg_eap = sentiments_eap[sentiments_eap$sentiment == 'Negative',]


#Word Clouds for EAP
par(mfrow=c(1,2))
wordcloud(sent_pos_eap$text,sent_pos_eap$freq, colors = brewer.pal(6, "Dark2") , min.freq=30,random.order = F)
text(x=0.5, y=1.05, "Positive")
wordcloud(sent_neg_eap$text,sent_neg_eap$freq, min.freq=20,colors=brewer.pal(6,"Dark2"),random.order = F)
text(x=0.5, y=1.05, "Negative")


#Creating a DTM  and sentiments for HPL


corpus = Corpus(VectorSource(list(hpl$text)))
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers) 
corpus = tm_map(corpus, stripWhitespace)
corpus = tm_map(corpus, removeWords, stopwords('english'))

dtm_hpl = DocumentTermMatrix(VCorpus(VectorSource(corpus[[1]]$content)))
freq_hpl <- colSums(as.matrix(dtm_hpl))

sentiments_hpl = calculate_sentiment(names(freq_hpl))
sentiments_hpl = cbind(sentiments_hpl, as.data.frame(freq_hpl))

sent_pos_hpl = sentiments_hpl[sentiments_hpl$sentiment == 'Positive',]
sent_neg_hpl = sentiments_hpl[sentiments_hpl$sentiment == 'Negative',]

#Word Cloud for HPL

par(mfrow=c(1,2))
wordcloud(sent_pos_hpl$text,sent_pos_hpl$freq, min.freq=30,colors=brewer.pal(6,"Dark2"),random.order = F)
text(x=0.5, y=1.05, "Positive")
wordcloud(sent_neg_hpl$text,sent_neg_hpl$freq, min.freq=20,colors=brewer.pal(6,"Dark2"),random.order = F)
text(x=0.5, y=1.05, "Negative")


##Creating a DTM  and sentiments for MWS
corpus = Corpus(VectorSource(list(mws$text)))
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers) 
corpus = tm_map(corpus, stripWhitespace)
corpus = tm_map(corpus, removeWords, stopwords('english'))

dtm_mws = DocumentTermMatrix(VCorpus(VectorSource(corpus[[1]]$content)))
freq_mws <- colSums(as.matrix(dtm_mws))

sentiments_mws = calculate_sentiment(names(freq_mws))
sentiments_mws = cbind(sentiments_mws, as.data.frame(freq_mws))

sent_pos_mws = sentiments_mws[sentiments_mws$sentiment == 'Positive',]
sent_neg_mws = sentiments_mws[sentiments_mws$sentiment == 'Negative',]



#Word Cloud for MWS
par(mfrow=c(1,2))
wordcloud(sent_pos_mws$text,sent_pos_mws$freq, min.freq=30,colors=brewer.pal(6,"Dark2"))
text(x=0.5, y=1.05, "Positive")
wordcloud(sent_neg_mws$text,sent_neg_mws$freq, min.freq=20,colors=brewer.pal(6,"Dark2"))
text(x=0.5, y=1.05, "Negative")



par(mfrow=c(1,1))


#Sentiment Plots
sentiments_Allan_Poe = cbind("Author"="Edgar_Allan_poe",sentiments_eap)
colnames(sentiments_Allan_Poe)[4] <- "freq"
sentiments_HP_Lovecraft = cbind("Author"="HP_Lovecraft",sentiments_hpl)
colnames(sentiments_HP_Lovecraft)[4] <- "freq"
sentiments_Mary = cbind("Author"="Mary_Wollstonecraft_Shelley",sentiments_mws)
colnames(sentiments_Mary)[4] <- "freq"

sentiments <- rbind(sentiments_Allan_Poe[,-2],sentiments_HP_Lovecraft[,-2],sentiments_Mary[,-2])
sentiments <- as.data.table(sentiments)
sentiments1 <- sentiments[,.N,by=.(sentiment,Author)]
sentiments1[,"Total":=sum(N),by=Author]
sentiments1 <- sentiments1[,.("Percentage"=100*N/Total),by=.(Author,sentiment)]


ggplot(sentiments1,aes(x = sentiment,y = Percentage ,fill=sentiment ))+
  geom_bar(stat = "identity") +
  ggtitle("Authors Sentiments")+xlab("Sentiment")+ylab("% Sentiment")+ 
  theme(axis.text.x = element_text(angle = 45, size=8,hjust = 1))+
  facet_wrap(~Author)







