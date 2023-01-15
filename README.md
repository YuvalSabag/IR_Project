# Information Retrieval - Search Engine
The final project for our "Information Retrieval" course at BGU involves creating a search engine for English Wikipedia. This search engine is able to process and index over six million documents, designed to quickly and efficiently return relevant results for user queries from the all corpus of Wikipedia articles.

### Retrival Methodes:
Our search engine utilizes a combination of various retrieval methods:
- Inverted Index
- TF-IDF
- BM-25
- Word2vac
- Cosine Similarity
- Page Rank
- Page view

### Indexes:
- Index_Title
- Index_Body
- Index_Anchor

### Capabillities:
#### Main Method Search-
Retrieve information using a query by utilizing both the body and title indexes (in a 0.5-0.5 ratio) using the BM-25 scoring algorithm. Additionally, we take into account the page rank and page view of the documents when determining the order of the results returned. we also use of WORD2VEC to enhance the relevance of the results.

##### Additionally, our search engine has the capability to retrieve information using 5 different techniques:

-Search body: retrive information only through the wiki page body. Use tf-idf and cosine similarity measure for comparrison, with a tf-idf thresh of over 0.45 per term.
-Serach title: retrive information only through the wiki page title. Use a binary ranking of terms existing or not in the title. More terms in title get priorotize.
-Search anchor: retrive information only through the wiki page title. Use a binary ranking of terms existing or not in the title. More terms in title get priorotize.
-Get pageview: retrive a specific wiki page amount of views.
-Get pagerank: retrive a specific wiki page rank.




