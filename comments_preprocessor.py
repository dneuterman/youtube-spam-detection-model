import re

from nltk.corpus import stopwords
import string

#function to remove stop words, punctuation, numbers and URLs from comment
def comments_preprocessor(comment):
    #regular expression attributed to https://urlregex.com/index.html
    http_urlhyperlink_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    comment = re.sub(http_urlhyperlink_regex, 'urlsubstitute', comment)

    #regular expression attributed to https://uibakery.io/regex-library/html-regex-python
    html_tags_regex = "<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>"
    comment = re.sub(html_tags_regex, '', comment)

    #removes \ufeff from comment
    comment = comment.replace("\ufeff", '')

    #removes punctuation
    clean_comment_array = []
    for char in comment:
        if char not in string.punctuation:
            clean_comment_array.append(char)

    clean_comment_array = ''.join(clean_comment_array).lower()
    clean_comment_array = clean_comment_array.split()

    #removes stop words. Isalpha removes numbers which also removes emjoi characters.
    word_list = []
    for word in clean_comment_array:
        if word.lower() not in stopwords.words('english') and word.isalpha():
            word_list.append(word)

    return word_list


