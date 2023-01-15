# Information Retrieval - Search Engine
The final project for our "Information Retrieval" course at BUG involves creating a search engine for English Wikipedia documents. This search engine is able to process and index over six million documents, designed to quickly and efficiently return relevant results for user queries from the all corpus of Wikipedia articles.

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
Through the engines end points, you can retrieve information using 5 different techniques:

-Search: retrive information with a query, use both body and title index (0.65-0.35 ratio of results favoring the body). Also include page rank and -page view consideration in return order.
-Search body: retrive information only through the wiki page body. Use tf-idf and cosine similarity measure for comparrison, with a tf-idf thresh of over 0.45 per term.
-Serach title: retrive information only through the wiki page title. Use a binary ranking of terms existing or not in the title. More terms in title get priorotize.
-Search anchor: retrive information only through the wiki page title. Use a binary ranking of terms existing or not in the title. More terms in title get priorotize.
-Get pageview: retrive a specific wiki page amount of views.
-Get pagerank: retrive a specific wiki page rank.




