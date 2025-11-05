import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import single_meteor_score
from pycocoevalcap.cider.cider import Cider
import bert_score
from typing import Union, Any, List, Optional, Literal
from transformers import PreTrainedTokenizerBase, AutoTokenizer

def load_tokenizer_model(tokenizer_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True, return_tensors="pt")
    return tokenizer
Tokenizer = load_tokenizer_model("<model_path>")

def tokenized_sentences(sentences: List[str], tokenizer: PreTrainedTokenizerBase=Tokenizer):
    tokens = tokenizer(sentences, add_special_tokens=False)["input_ids"]
    words = [[tokenizer.decode(i) for i in j] for j in tokens]
    return words

def calculate_bleu(reference: str, hypothesis: str, n: int=4) -> float:
    ref, hyp = tokenized_sentences([reference, hypothesis])
    weights = tuple()
    if n == 1:
        weights = (1, 0, 0, 0)
    elif n == 2:
        weights = (0.5, 0.5, 0, 0)
    elif n == 3:
        weights = (0.33, 0.33, 0.33, 0)
    elif n == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    else:
        return None
    return sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method1, weights=weights)

def get_bleu(references: List[str], hypothesises: List[str], n: int=4):
    if len(references) != len(hypothesises):
        raise ValueError('The length of values must be same!')
    num_samples = len(references)
    scores = 0.0
    for idx, (ref, hyp) in enumerate(zip(references, hypothesises)):
        scores+=calculate_bleu(ref, hyp, n)
    return scores / num_samples


def calculate_rouge(reference: str, hypothesis: str):
    return Rouge().get_scores(hypothesis, reference, avg=False)[0]

    
def calculate_bertscore(reference: str, hypothesis: str, language: Optional[Literal['en', 'zh']]='en'):
    lang = language.lower()
    if lang == 'en':
        P, R, F1 = bert_score.score(hypothesis, reference, lang=language, verbose=False, model_type="bert-base-uncased")
    elif lang == 'zh' or lang == 'zh-cn':
        P, R, F1 = bert_score.score(hypothesis, reference, lang=language, verbose=False, model_type="bert-base-chinese")
    else:
        P, R, F1 = bert_score.score(hypothesis, reference, lang=language, verbose=False, model_type="google-bert/bert-base-multilingual-cased")
    return P.mean().item(), R.mean().item(), F1.mean().item()

def calculate_meteor(reference: str, hypothesis: str):
    ref, hyp = tokenized_sentences([reference, hypothesis])
    return single_meteor_score(ref, hyp)

def get_rouge(references: List[str], hypothesises: List[str]):
    if len(references) != len(hypothesises):
        raise ValueError('The length of values must be same!')
    return Rouge().get_scores(hypothesises, references, avg=True)

def get_meteor(references: List[str], hypothesises: List[str]):
    if len(references) != len(hypothesises):
        raise ValueError('The length of values must be same!')
    num_samples = len(references)
    scores = 0.0
    for idx, (ref, hyp) in enumerate(zip(references, hypothesises)):
        scores+=calculate_meteor(ref, hyp)
    return scores / num_samples

def get_cider(references: List[str], hypothesises: List[str]):
    if len(references) != len(hypothesises):
        raise ValueError('The length of values must be same!')
    refs, tgts = {}, {}
    for idx, (ref, tgt) in enumerate(zip(references, hypothesises)):
        refs[f'{idx}'] = [ref]
        tgts[f'{idx}'] = [tgt]
    return Cider().compute_score(refs, tgts)[0] / 10

def get_acc(references: List[Any], hypothesises: List[Any]):
    if len(references) != len(hypothesises):
        raise ValueError('The length of values must be same!')
    total_len = len(references)
    right = 0
    for ref, hyp in zip(references, hypothesises):
        if ref == hyp:
            right+=1
    return right / total_len


# print(Tokenizer(["I love you", "我爱中国"], add_special_tokens=True))
# print(Tokenizer.eos_token_id)