import bs4
import re
import unicodedata


def strip_accents(text):

        try:
            text = unicode(text, 'utf-8')
        except NameError: # unicode is a default on python 3 
            pass
        
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        
        return str(text)


def tweet_cleaner(text):
    
    #Change to all lower case
    text = text.lower()

    #task 1: Clean up HTML (mainly new lines)
    soup = bs4.BeautifulSoup(text, 'lxml')
    text = soup.get_text()
    text = text.replace('\n',' ')
    text = text.replace('\r','.')
    
    #task 2: Find if RT and remove
    text = re.sub(r'^rt @[A-Za-z0-9_]+:','', text)

    #task 3: Remove mentions
    text = re.sub(r'@[A-Za-z0-9_]+','',text)

    #task 4: remove links
    text = re.sub('https?://[A-Za-z0-9./~]+','', text)
    
    #Change question and exclamation marks to full stops to split into sentences more easily (might not use)
    text = re.sub('[?!]','.', text)
    
    #Remove ellipsis and multiple dots turning them into full stops
    text = re.sub('\u2026','.', text)
    text = re.sub('(\.\s?)+','. ', text)
    
    #All characters repeated 2 times or more replaced with two characters
    text = re.sub(r'(\S)\1{3,}',r'\1\1', text)
    
    #Numbers with monetary value to monetary symbol
    text = re.sub(r'[$£€]\d+(\.\d+)?|[^\w]\d+(\.\d+)?[$£€]',r' moneyvaluesymbol', text)
    
    #Percentages
    text = re.sub(r'\s\d+(\.\d+)?%',r' percentagesymbol ', text)
    
    #Times
    text = re.sub('\s\d+:\d+\s',r' colonnumbersymbol ',text)
    
    #Dates
    text = re.sub(r'\s\d{1,2}st|\s\d{1,2}th',' datesymbol',text)
    
    #Phone numbers
    text = re.sub('(\d+-\d+)+',r' dashednumbersymbol ',text)
    
    #Standalone Numbers to number symbol
    text = re.sub(r'(?<=[^$£€\w])\d+(\.\d+)?(?=[^$£€\w])|(?<=[^$£€\w])\d+(,\d+)?(?=[^$£€\w])',r' numbersymbol ', text)

    #task 5: capture hashtags, (don't) store them and remove from text
    #re.findall(r'#[a-zA-Z]',text)
    text = re.sub(r'(#\w+)','hashtagsymbol ',text)
    
    #special meanings
    #ampersand code
    text = re.sub(r'&amp;','@',text)
    #left arrow code
    text = re.sub(r'&lt;','<',text)
    #right arrow code
    text = re.sub(r'&gt;','>',text)
    #quote marks
    text = re.sub(r'&quot;',' quotesymbol ',text)

    #task 6: remove numbers and other characters such as colons from the ends of non-emojis
    #colon from the end of proper words or numbers
    text = re.sub(r'(^|\s)([A-z09-9\.,]+):(\s|$)',r' \2 ', text)
    #ampersand from the end of proper words or numbers
    text = re.sub(r'(^|\s)([A-z09-9\.,]+)@(\s|$)',r' \2 ', text)
    #brackets
    text = re.sub(r'[\(\)]+([\w]+)[\(\)]+',r' \1 ',text)
    text = re.sub(r'[\(\)](\w+)|(\w+)[\(\)]',r' \1 \2 ',text)
    #Stars at the end or start of proper words/numbers
    text = re.sub(r'(^|\s)([A-z09-9\.,]+)\*(\s|$)',r' \2 starsymbol ', text)
    text = re.sub(r'(^|\s)\*([A-z09-9\.,]+)(\s|$)',r' starsymbol \2 ', text)
    #Pipes at the start of ends of proper wors/numbers
    text = re.sub(r'(^|\s)([A-z09-9\.,]+)\|(\s|$)',r' \2 ', text)
    text = re.sub(r'(^|\s)\|([A-z09-9\.,]+)(\s|$)',r' \2 ', text)
    #too many special characters in a row
    text = re.sub(r'[\-\\\:/@&\*;\^~#%$!<>]{3,}','specialstringsymbol ',text)
    #special characters in the wrong place
    #special characters on their own
    text = re.sub(r'(?<=[\s\d])[\-\\:/@&\*;\^~#%><]+(?=[\s\d])',' ',text)
    #special characters on their own at the start or end of text
    text = re.sub(r'^[\-\\:/@&\*;\^~#%><]+(?=[\s\d])|(?<=[\s\d])[\-\\:/@&\*;\^~#%><]+$',' ',text)
    
    
    #Quote marks
    text = re.sub(r'"',r' quotesymbol ',text)
    
    #remove the non-standard hyphen special characters
    text = re.sub(r'[\u2010-\u2015\u200a]',' ', text, flags = re.UNICODE)
#     text = re.sub(r'[.!]', ' ', text, flags = re.UNICODE)
    
    #task 7: remove unicode emojis
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r' ', text)
    
    # Find truncated words on ends of others
    #negations
    text =re.sub("(?<=[A-z])n't\s", " n't ", text)
    #be
    text =re.sub("(?<=it)'s\s", " is ", text)
    #will
    text =re.sub("(?<=[A-z])'ll\s", " will ", text)
    #would
    text =re.sub("(?<=[A-z])'d\s", " would ", text)
    #have
    text =re.sub("(?<=[A-z])'ve\s", " have ", text)
    #are
    text =re.sub("(?<=[A-z])'re\s", " are ", text)
    #i'm
    text =re.sub("(?<=i)'m\s", " am ", text)
    
    #Belongs to
    text =re.sub("(?<=[A-z])'s\s", " <belongstosymbol> ", text)
    
    #Any more times or dates
    text =re.sub("[0-9]+?(a|p)m(\s|$)", " timesymbol ", text)
    text =re.sub("[0-9]+?(st|th)(\s|$)", " posordatesymbol ", text)
    
    
    #Change accented characters to closest non-accented translation
    text = strip_accents(text)

    #Lemmatization to be done at end
    
    return text
