import nltk
from nltk.tokenize import (
    word_tokenize, sent_tokenize, blankline_tokenize,
    WhitespaceTokenizer, wordpunct_tokenize
)
from nltk.util import bigrams, trigrams, ngrams
from nltk.stem import (
    PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
)
from nltk import pos_tag, ne_chunk
from nltk.corpus import stopwords

# Optional: nltk.download() if you're running for the first time

AI1 = '''In recent months, Jammu and Kashmir has witnessed a series of terrorist
attacks targeting security forces, civilians, and government infrastructure.
These attacks have raised concerns about the security situation in the region,
as militants continue to carry out violence despite ongoing counter-terrorism
efforts by Indian security forces. The attacks often involve ambushes, grenade
explosions, and targeted killings, leading to casualties on both sides.
The government has responded with increased security measures and operations
aimed at neutralizing militant groups, but the situation remains tense.
These incidents have also fueled debates about the effectiveness of the security
approach in the region and the ongoing challenge of restoring peace and stability.
The impact of such attacks is not only felt in terms of human lives but also on
the local economy and the daily lives of people in the affected areas.'''

print("Length of paragraph:", len(AI1))

# Word Tokenization
word_tokens = word_tokenize(AI1)
print("Word tokens:", word_tokens)
print("Total words:", len(word_tokens))

# Sentence Tokenization
sent_tokens = sent_tokenize(AI1)
print("Sentence tokens:", sent_tokens)
print("Total sentences:", len(sent_tokens))

# Blankline Tokenization
blank_tokens = blankline_tokenize(AI1)
print("Blankline tokens:", blank_tokens)

# Whitespace Tokenization
whitespace_tokens = WhitespaceTokenizer().tokenize(AI1)
print("Whitespace tokens:", whitespace_tokens)
print("Total whitespace tokens:", len(whitespace_tokens))

# Wordpunct Tokenization
wordpunct_tokens = wordpunct_tokenize(AI1)
print("Wordpunct tokens:", wordpunct_tokens)
print("Total wordpunct tokens:", len(wordpunct_tokens))

# Bigrams, Trigrams, Ngrams
string1 = "The 2025 IPL season is underway, with teams like Mumbai Indians and Chennai Super Kings battling for supremacy."
ipl_tokens = word_tokenize(string1)
print("Bigrams:", list(bigrams(ipl_tokens)))
print("Trigrams:", list(trigrams(ipl_tokens)))
print("4-grams:", list(ngrams(ipl_tokens, 4)))

# Stemming
pst = PorterStemmer()
lst = LancasterStemmer()
sbst = SnowballStemmer('english')

words_to_stem = ['give', 'giving', 'given', 'gaved', 'thinking', 'loving', 'maximum']

print("\nPorterStemmer:")
for word in words_to_stem:
    print(word + " → " + pst.stem(word))

print("\nLancasterStemmer:")
for word in words_to_stem:
    print(word + " → " + lst.stem(word))

print("\nSnowballStemmer:")
for word in words_to_stem:
    print(word + " → " + sbst.stem(word))

# Lemmatization
lemmatizer = WordNetLemmatizer()
print("\nLemmatization:")
for word in words_to_stem:
    print(word + " → " + lemmatizer.lemmatize(word))

# POS Tagging
sent = 'sam is a natureal when it comes to drawing'
tokens = word_tokenize(sent)
print("\nPOS Tags:")
for token in tokens:
    print(pos_tag([token]))

# Another POS Example
sent1 = 'aksh is machine larning is very pefect and good knowlege'
sent1_tokens = word_tokenize(sent1)
print("\nPOS Tags for sent1:")
for token in sent1_tokens:
    print(pos_tag([token]))

# Named Entity Recognition
NE_sent = 'The Indain prime minester stay in the Delhli'
NE_tokens = word_tokenize(NE_sent)
NE_tags = pos_tag(NE_tokens)
NE_tree = ne_chunk(NE_tags)
print("\nNamed Entity Recognition:")
print(NE_tree)

# Stopwords
print("\nEnglish Stopwords sample:")
print(stopwords.words('english')[:20])
print("Total English stopwords:", len(stopwords.words('english')))

print("\nFrench Stopwords sample:")
print(stopwords.words('french')[:20])
print("Total French stopwords:", len(stopwords.words('french')))

print("\nGerman Stopwords sample:")
print(stopwords.words('german')[:20])
print("Total German stopwords:", len(stopwords.words('german')))

print("\nChinese Stopwords sample:")
print(stopwords.words('chinese')[:20])
print("Total Chinese stopwords:", len(stopwords.words('chinese')))
