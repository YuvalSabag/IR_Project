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

#### Additionally, our search engine has the capability to retrieve information using 5 different techniques- 

- Search body: Retrieve information exclusively from the Wikipedia page bodies by utilizing cosine similarity as the primary method of comparison.
- Serach title: Retrieve information by exclusively examining the titles of Wikipedia pages through a binary ranking system that gives priority to articles with a greater number of query terms present in their title.
- Search anchor: Retrieve information by exclusively examining the Wikipedia anchor text pages through a binary ranking system that gives priority to articles with a greater number of query terms present in anchor text.
- Get page rank: Retrieve the PageRank score of a specific Wikipedia article identified by its unique ID, based on the internal links within the article.
- Get page view: Retrieve the number of page views for a specific Wikipedia article, identified by its unique article ID.



