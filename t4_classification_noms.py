# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import glob
import os
import string
import unicodedata
import json

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

datafiles = "./data/names/*.txt"  # les fichiers pour construire vos modèles
test_filename = './data/test_names.txt'  # le fichier contenant les données de test pour évaluer vos modèles

names_by_origin = {}  # un dictionnaire qui contient une liste de noms pour chaque langue d'origine
all_origins = []  # la liste des 18 langues d'origines de noms 

# Fonctions utilitaires pour lire les données d'entraînement et de test - NE PAS MODIFIER

def load_names():
    """Lecture des noms et langues d'origine d'un fichier. Par la suite,
       sauvegarde des noms pour chaque origine dans le dictionnaire names_by_origin."""
    for filename in find_files(datafiles):
        origin = get_origin_from_filename(filename)
        all_origins.append(origin)
        names = read_names(filename)
        names_by_origin[origin] = names
        

def find_files(path):
    """Retourne le nom des fichiers contenus dans un répertoire.
       glob fait le matching du nom de fichier avec un pattern - par ex. *.txt"""
    return glob.glob(path)


def get_origin_from_filename(filename):
    """Passe-passe qui retourne la langue d'origine d'un nom de fichier.
       Par ex. cette fonction retourne Arabic pour "./data/names/Arabic.txt". """
    return os.path.splitext(os.path.basename(filename))[0]


def read_names(filename):
    """Retourne une liste de tous les noms contenus dans un fichier."""
    with open(filename, encoding='utf-8') as f:
        names = f.read().strip().split('\n')
    return [unicode_to_ascii(name) for name in names]


def unicode_to_ascii(s):
    """Convertion des caractères spéciaux en ascii. Par exemple, Hélène devient Helene.
       Tiré d'un exemple de Pytorch. """
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def load_test_names(filename):
    """Retourne un dictionnaire contenant les données à utiliser pour évaluer vos modèles.
       Le dictionnaire contient une liste de noms (valeurs) et leur origine (clé)."""
    with open(filename, 'r') as fp:
        test_data = json.load(fp)
    return test_data

#---------------------------------------------------------------------------
# Fonctions à développer pour ce travail - Ne pas modifier les signatures et les valeurs de retour

def cross_vallidation(classifier, X, y, n):
  scores = cross_val_score(classifier, X, y, cv=5)
  print('\n ****** Type de classificateur :', classifier, ' n gramme = ', n)
  print('\t Évaluation par validation croisée (en entrainement) : ')
  print('\tAccuracy sur chaque partition ', scores)
  print('\tExactitude moyenne: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2))

def train_NB(X, y, n):
  model = MultinomialNB()
  model.fit(X, y)
    #Eval sur train
  cross_vallidation(model, X, y, n)
  return model

def train_LR(X, y, n):
  model = LogisticRegression(max_iter=400)
  model.fit(X, y)
    #Eval sur train
  cross_vallidation(model, X, y, n)
  return model

def train_classifiers():
    load_names()
    # Vous ajoutez à partir d'ici tout le code dont vous avez besoin
    # pour construire les différentes versions de classificateurs de langues d'origines.
    # Voir les consignes de l'énoncé du travail pratique pour déterminer les différents modèles à entraîner.
    #
    # On suppose que les données d'entraînement ont été lues (load_names) et sont disponibles (names_by_origin).
    #
    # Vous pouvez ajouter au fichier toutes les fonctions que vous jugerez nécessaire.
    # Assurez-vous de sauvegarder vos modèles pour y accéder avec la fonction get_classifier().
    # On veut éviter de les reconstruire à chaque appel de cette fonction.
    # Merci de ne pas modifier les signatures (noms de fonctions et arguments) déjà présentes dans le fichier.
    #
    # Votre code à partir d'ici...
    #
    classes = all_origins
    X_train, y_train = [], []
    for origin_ in classes:
      for i in range(len(names_by_origin[origin_])):
        X_train.append(names_by_origin[origin_][i])
        y_train.append(origin_)

    unigram_vectorizer_char = CountVectorizer(analyzer='char', lowercase=True, ngram_range=(1,1))
    unigram_vectorizer_char.fit(X_train)
    print('\n Number d\'attributs de classification: ', len(unigram_vectorizer_char.get_feature_names()))
    X_uni_vec = unigram_vectorizer_char.transform(X_train)

    # Naive Bayes 1-gram
    model_uni_NB = train_NB(X_uni_vec, y_train, 1)
    # Logistic regression 1-gram
    model_uni_LR = train_LR(X_uni_vec, y_train, 1)

    bigram_vectorizer_char = CountVectorizer(analyzer='char', lowercase=True, ngram_range=(2,2))
    bigram_vectorizer_char.fit(X_train)
    X_bi_vec = bigram_vectorizer_char.transform(X_train)

    # Naive Bayes 2-gram
    model_bi_NB = train_NB(X_bi_vec, y_train, 2)
    # Logistic regression 2-gram
    model_bi_LR = train_LR(X_bi_vec, y_train, 2)

    trigram_vectorizer_char = CountVectorizer(analyzer='char', lowercase=True, ngram_range=(3,3))
    trigram_vectorizer_char.fit(X_train)
    X_tri_vec = trigram_vectorizer_char.transform(X_train)

    # Naive Bayes 3-gram
    model_tri_NB = train_NB(X_tri_vec, y_train, 3)
    # Logistic regression 3-gram
    model_tri_LR = train_LR(X_tri_vec, y_train, 3)

    multigram_vectorizer_char = CountVectorizer(analyzer='char', lowercase=True, ngram_range=(1,3))
    multigram_vectorizer_char.fit(X_train)
    X_mlti_vec = multigram_vectorizer_char.fit_transform(X_train)

    # Naive Bayes Multi-gram
    model_multi_NB = train_NB(X_mlti_vec, y_train, 'multi')
    # Logistic regression 3-gram
    model_multi_LR = train_LR(X_mlti_vec, y_train, 'multi')

    return ({'NB': [model_uni_NB, model_bi_NB, model_tri_NB, model_multi_NB],
            'LR': [model_uni_LR, model_bi_LR, model_tri_LR, model_multi_LR]},
            [unigram_vectorizer_char, bigram_vectorizer_char,
             trigram_vectorizer_char, multigram_vectorizer_char])
    
def get_classifier(type, n=3):
    # Retourne le classificateur correspondant. On peut appeler cette fonction
    # après que les modèles ont été entraînés avec la fonction train_classifiers
    #
    # type = 'NB' pour naive bayes ou 'LR' pour régression logistique
    # n = 1,2,3 ou multi
    #

    # À modifier
    if type == 'NB':
        return MultinomialNB()
    elif type == 'LR':
        return LogisticRegression()
    else:
        raise ValueError("Unknown model type")

#Variable global
model_dict = train_classifiers() 

def origin(name, type, n=3):
    # Retourne la langue d'origine prédite pour le nom.
    #   - name = le nom qu'on veut classifier
    #   - type = 'NB' pour naive bayes ou 'LR' pour régression logistique
    #   - n désigne la longueur des N-grammes de caractères. Choix possible = 1, 2, 3, 'multi'
    #
    # Votre code à partir d'ici...
    # À compléter...
    #
    if n == 2:
      name_vectorized = model_dict[1][3].transform([name])
      name_origin = model_dict[0][type][3].predict(name_vectorized)
    else:
      name_vectorized = model_dict[1][n-1].transform([name])
      name_origin = model_dict[0][type][n-1].predict(name_vectorized)
    return name_origin 
    
    
def evaluate_classifier(test_fn, type, n=3):
    test_data = load_test_names(test_fn)

    # Insérer ici votre code pour la classification des noms.
    # Votre code...

    classes = test_data.keys()
    X_test, y_test = [], []
    for origin_ in classes:
      for i in range(len(test_data[origin_])):
        X_test.append(test_data[origin_][i])
        y_test.append(origin_)
    if n == 'multi':
      test_vectorized = model_dict[1][3].transform(X_test)
      y_pred = model_dict[0][type][3].predict(test_vectorized)
    else:
      test_vectorized = model_dict[1][n-1].transform(X_test)
      y_pred = model_dict[0][type][n-1].predict(test_vectorized)

    test_accuracy = accuracy_score(y_test, y_pred)
    return test_accuracy


if __name__ == '__main__':
    # Vous pouvez modifier cette section comme bon vous semble
    print("Les {} langues d'origine sont: \n{}".format(len(all_origins), all_origins))
    chinese_names = names_by_origin["Chinese"]
    print("\nQuelques noms chinois : \n", chinese_names[:20])

    model = 'LR'
    ngram_length = 1

    classifier = get_classifier(model, n=ngram_length)
    print("\nType de classificateur: ", classifier)

    some_name = "Lamontagne"
    some_origin = origin(some_name, model, n=ngram_length)
    print("\nLangue d'origine de {}: {}".format(some_name, some_origin))

    test_names = load_test_names(test_filename)
    print("\nLes données pour tester vos modèles sont:")
    for org, name_list in test_names.items():
        print("\t{} : {}".format(org, name_list))
    evaluation = evaluate_classifier(test_filename, model, n=ngram_length)
    print("\nAccuracy_test= {}".format(evaluation))
