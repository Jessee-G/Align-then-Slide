import spacy
from constant import *


nlp_en = spacy.load("en_core_web_sm")
nlp_zh = spacy.load("zh_core_web_sm")

spacy_map_list = {
    ZH: nlp_zh,
    EN: nlp_en,
}


def sep_doc(long_text, lang):
    '''
    Sentence Pre-Segmentation
    :param long_text:
    :param lang:
    :return:
    '''
    def sep_para(para, model):
        doc_deal = model(para)
        sents = [_.text.strip() for _ in doc_deal.sents]
        sents = [_ for _ in sents if len(_) > 0]
        return sents

    spacy_nlp_model = spacy_map_list[lang]

    final_sents = []
    for para in long_text.split("\n"):
        para = para.strip()
        if len(para) > 0:
            final_sents += sep_para(para, spacy_nlp_model)

    return final_sents
