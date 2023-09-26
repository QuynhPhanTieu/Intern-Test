from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from keybert import KeyBERT


sentence_model = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')
kw_model = KeyBERT(model = sentence_model)


class KeywordExtraction:
    def __init__(self, text):
        self.text = text


    def extract(self):

        #Tokenize text
        text_tokenized = tokenize(self.text)
        text_splitted = text_tokenized.split(' ')
        
        #Remove Vietnamese stopwords
        with open('vietnames-stopwords.txt', encoding="utf8") as stop_fil:
            stopwords = set(stop_fil.read().lower().split("\n"))

        stopwords = list(stopwords)

        sw_tokenized = [tokenize(sw) for sw in stopwords]

        text_without_sw = [word for word in text_splitted if not word in sw_tokenized]

        text_cleaned = ' '.join(text_without_sw)
   
        keywords = kw_model.extract_keywords(text_cleaned)     

        return keywords
    


text = "Không gian địa lý của vùng Tây Bắc hiện còn chưa được nhất trí. Một số ý kiến cho rằng đây là vùng phía nam (hữu ngạn) sông Hồng. Một số ý kiến lại cho rằng đây là vùng phía nam của dãy núi Hoàng Liên Sơn. Nhà địa lý học Lê Bá Thảo cho rằng vùng Tây Bắc được giới hạn ở phía đông bởi dãy núi Hoàng Liên Sơn và ở phía tây là dòng sông Mã."
keyword_extraction = KeywordExtraction(text)
keywords = keyword_extraction.extract()

print(keywords)