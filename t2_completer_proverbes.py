import json
from nltk import word_tokenize, bigrams, trigrams, pad_sequence 
from nltk.util import ngrams
from nltk.lm.models import Laplace

proverbs_fn = "./data/proverbes.txt"
test1_fn = "./data/test_proverbes.txt"
BOS = '<BOS>'
EOS = '<EOS>'

def build_vocabulary(text_list):
    all_unigrams = list()
    for sentence in text_list:
        word_list = word_tokenize(sentence.lower())
        all_unigrams = all_unigrams + word_list
    voc = set(all_unigrams)
    voc.add(BOS)
    voc.add(EOS)
    return list(voc)

def get_ngrams(text_list, n=2):
    all_ngrams = list()
    for sentence in text_list:
        tokens = word_tokenize(sentence.lower())
        padded_sent = list(pad_sequence(tokens, pad_left=True, left_pad_symbol=BOS, pad_right=True, right_pad_symbol=EOS, n=n))
        all_ngrams = all_ngrams + list(ngrams(padded_sent, n=n))      
    return all_ngrams

def load_proverbs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    return [x.strip() for x in raw_lines]


def load_tests(filename):
    with open(filename, 'r', encoding="utf-8") as fp:
        test_data = json.load(fp)
    return test_data

def train_model_n(tokens, proverbs, order):
    n_gram = get_ngrams(proverbs, order)
    model = Laplace(order)
    model.fit([n_gram], vocabulary_text=tokens)
    return model

def train_models(filename):
    proverbs = load_proverbs(filename)
    """ Vous ajoutez à partir d'ici tout le code dont vous avez besoin
        pour construire les différents modèles N-grammes.
        Voir les consignes de l'énoncé du travail pratique concernant les modèles à entraîner.

        Vous pouvez ajouter au fichier les classes, fonctions/méthodes et variables que vous jugerez nécessaire.
        Il faut au minimum prévoir une variable (par exemple un dictionnaire) 
        pour conserver les modèles de langue N-grammes après leur construction. 
        Merci de ne pas modifier les signatures (noms de fonctions et arguments) déjà présentes dans le fichier.
    """
    tokens = build_vocabulary(proverbs)
    model_list = [train_model_n(tokens, proverbs, x) for x in range(1, 4)]
    return model_list

model_list = train_models(proverbs_fn)

def cloze_test(incomplete_proverb, choices, n=3, criteria="perplexity"):
    """ Fonction qui complète un texte à trous (des mots masqués) en ajoutant le bon mot.
        En anglais, on nomme ce type de tâche un "cloze test".

        Le paramètre criteria indique la mesure qu'on utilise pour choisir le mot le plus probable: "logprob" ou "perplexity".
        La valeur retournée est l'estimation sur le proverbe complet (c.-à-d. toute la séquence de mots du proverbe).

        Le paramètre n désigne le modèle utilisé.
        1 - unigramme NLTK, 2 - bigramme NLTK, 3 - trigramme NLTK
    """

    # Votre code à partir d'ici.Vous pouvez modifier comme bon vous semble.
    model = model_list[n-1]
    scores = []
    for valeur in choices:
      new_sentence = incomplete_proverb.replace('***', valeur)
      tokens = word_tokenize(new_sentence.lower())
      if n == 1:
        padded_sent = list(pad_sequence(tokens, pad_left=True, left_pad_symbol=BOS, pad_right=True, right_pad_symbol=EOS, n=1))
        if criteria == "perplexity":
          score = model.perplexity(padded_sent)
        else:
          score = model.logscore(valeur)
      if n == 2:
        padded_sent = list(pad_sequence(tokens, pad_left=True, left_pad_symbol=BOS, pad_right=True, right_pad_symbol=EOS, n=2))
        if criteria == "perplexity":
          bigrams_ = list(bigrams(padded_sent))
          score = model.perplexity(bigrams_)
        else:
          idx_valeur = padded_sent.index(valeur)
          score1 = model.logscore(valeur, [padded_sent[idx_valeur - 1]]) # qui vient
          score2 = model.logscore(padded_sent[idx_valeur + 1], [valeur]) # vient de
          score = score1 + score2
      if n == 3:
        padded_sent = list(pad_sequence(tokens, pad_left=True, left_pad_symbol=BOS, pad_right=True, right_pad_symbol=EOS, n=3))
        if criteria == "perplexity":
          trigrams_ = list(trigrams(padded_sent))
          score = model.perplexity(trigrams_)
        else:
          idx_valeur = padded_sent.index(valeur)
          score1 = model.logscore(valeur, [padded_sent[idx_valeur - 2], padded_sent[idx_valeur - 1]]) # mentir qui vient
          score2 = model.logscore(padded_sent[idx_valeur + 1], [padded_sent[idx_valeur - 1], valeur]) # qui vient de
          score3 = model.logscore(padded_sent[idx_valeur + 2], [valeur, padded_sent[idx_valeur + 1]]) # vient de loin
          score = score1 + score2 + score3
      scores.append([score, new_sentence])

    if criteria == "perplexity":
        score_f = min(scores)[0]
        result = min(scores)[1]
    else:
        score_f = max(scores)[0]
        result = max(scores)[1]
    return result, score_f


if __name__ == '__main__':
    # Vous pouvez modifier cette section comme bon vous semble
    proverbs = load_proverbs(proverbs_fn)
    print("\nNombre de proverbes pour entraîner les modèles : ", len(proverbs))
    train_models(proverbs_fn)

    test_proverbs = load_tests(test1_fn)
    print("\nNombre de tests du fichier {}: {}\n".format(test1_fn, len(test_proverbs)))
    print("Les résultats des tests sont:")
    for partial_proverb, options in test_proverbs.items():
        solution, valeur = cloze_test(partial_proverb, options, n=3, criteria="logprob")
        print("\n\tProverbe incomplet: {} , Options: {}".format(partial_proverb, options))
        print("\tSolution = {} , Valeur = {}".format(solution, valeur))
