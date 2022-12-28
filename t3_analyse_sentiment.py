# -*- coding: utf-8 -*-
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
import spacy
import numpy as np

analyzer_en = spacy.load("en_core_web_sm")

reviews_dataset = {
    'train_pos_fn': "./data/senti_train_positive.txt",
    'train_neg_fn': "./data/senti_train_negative.txt",
    'test_pos_fn': "./data/senti_test_positive.txt",
    'test_neg_fn': "./data/senti_test_negative.txt"
}


def load_reviews(filename):
    with open(filename, 'r') as fp:
        reviews_list = json.load(fp)
    return reviews_list


def prepare_dataset(dataset):
    partitions = []
    splits = ['train_pos_fn', 'train_neg_fn', 'test_pos_fn', 'test_neg_fn']
    for split in splits:
        partitions.append(load_reviews(reviews_dataset[split]))
    value = 0
    for i in range(len(partitions)):
        for j in range(len(partitions[i])):
            partitions[i][j] = [partitions[i][j], value % 2]
        value += 1
    train_array = np.array(partitions[0] + partitions[1])
    test_array = np.array(partitions[2] + partitions[3])
    return train_array, test_array


def normalise(sentence, normalisation="words"):
    if normalisation == "words":
        return sentence
    if normalisation == "stem":
        stemmer = PorterStemmer()
        tokens = word_tokenize(sentence)
        sentence = " ".join([stemmer.stem(word) for word in tokens])
        return sentence
    if normalisation == "lemma":
        tokens = analyzer_en(str(sentence))
        result = " ".join(token.lemma_ for token in tokens)
        return result


def evaluate_classfier(classifier, x, y):
    y_pred = classifier.predict(x)
    return accuracy_score(y, y_pred), y_pred


def train_and_test_classifier(dataset, model='NB', normalization='words'):
    """
    :param dataset: un dictionnaire contenant le nom des 4 fichiers utilisées pour entraîner et tester les classificateurs. Voir variable reviews_dataset.
    :param model: le type de classificateur. NB = Naive Bayes, LR = Régression logistique.
    :param normalization: le prétraitement appliqué aux mots des critiques (reviews)
                 - 'word': les mots des textes sans normalization.
                 - 'stem': les racines des mots obtenues par stemming.
                 - 'lemma': les lemmes des mots obtenus par lemmatisation.
    :return: un dictionnaire contenant 3 valeurs:
                 - l'accuracy à l'entraînement (validation croisée)
                 - l'accuracy sur le jeu de test
                 - la matrice de confusion calculée par scikit-learn sur les données de test
    """
    train, test = prepare_dataset(dataset)
    vectorizer = CountVectorizer(lowercase=True)
    train_x_vectorizer = vectorizer.fit_transform(
        [normalise(sentence, normalization) for sentence in train[:, 0]])
    test_x_vectorizer = vectorizer.transform(
        [normalise(sentence, normalization) for sentence in test[:, 0]])
    X_train, X_test, y_train, y_test = train_x_vectorizer, test_x_vectorizer, train[
        :, 1], test[:, 1]

    if model == "NB":
        classifier = MultinomialNB()
    if model == "LR":
        classifier = LogisticRegression(max_iter=100000)

    classifier.fit(X_train, y_train)
    # Les résultats à retourner
    results = dict()
    results['accuracy_train'] = evaluate_classfier(
        classifier, X_train, y_train)[0]
    acc, y_pred = evaluate_classfier(classifier, X_test, y_test)
    results['accuracy_test'] = acc
    # la matrice de confusion obtenue de Scikit-learn
    results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    return results


if __name__ == '__main__':
    results = train_and_test_classifier(
        reviews_dataset, model='LR', normalization='words')
    print("Accuracy - entraînement: ", results['accuracy_train'])
    print("Accuracy - test: ", results['accuracy_test'])
    print("Matrice de confusion: ", results['confusion_matrix'])
