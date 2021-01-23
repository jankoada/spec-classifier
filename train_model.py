#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight
from collections import Counter
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

import config


pattern_end = '[\.\:,!?\n\r\t\s]'
pattern_suffix = pattern_end + '*)'
patterns = []
# Template sentences
patterns.append('(Co vás trápí\? Jak se potíže projevují a jak dlouho trvají\.\:' + pattern_suffix)
patterns.append('(Jak vám můžeme pomoci\? Na co se našich lékařů chcete zeptat\?\:' + pattern_suffix)
patterns.append(
    '(Doplňující informace\. Užívané léky, absolvovaná vyšetření, výška, váha, další onemocnění\:' + pattern_suffix)
patterns.append('(Co se chceš od lékaře dozvědět\?\:' + pattern_suffix)
patterns.append('(Jak dlouho problém trvá\?\:' + pattern_suffix)
patterns.append('(Léčíš se s nějakým závažným onemocněním\? Jakým\?\:' + pattern_suffix)
patterns.append('(Užíváš pravidelně nějaké léky\, hormony nebo antikoncepci\?\:' + pattern_suffix)
patterns.append('(Máš alergii na nějaké léky\?\:' + pattern_suffix)
patterns.append('(Máš představu, čeho se problém týká\?\:' + pattern_suffix)
patterns.append('(Doplňující dotaz\:' + pattern_suffix)
patterns.append('(What brings you here today\?\:' + pattern_suffix)
patterns.append('(Are you taking any medication\?\:' + pattern_suffix)
patterns.append('(Doplňující inforace\?\:' + pattern_suffix)
patterns.append('(Chceš se zeptat na své zdraví\?\:' + pattern_suffix)
patterns.append('(Co by měl lékař vědět o zdravotním problému\? Přilož fotku či lékařskou zprávu, pokud ji máš\.\?\:'
                + pattern_suffix)
pattern = '|'.join(patterns)


# Remove template sentences from questions
def preprocess_question(text):
    text = re.sub(pattern, '', text)
    return text


def clean_dataset(df):
    # Remove test data (questions containing string "*test*")
    df = df[~(df['question'].str.contains("\*test\*"))].copy()
    # Remove template sentences from questions
    df.loc[:, 'question'] = df.question.apply(preprocess_question)
    return df


def timestamp(label):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f'{current_time} {label}')


def load_dataset(path):
    df = pd.read_csv(path)
    df.question = df.question.astype('string')
    return df


def save_to_pickle(data, pickle_name):
    pickle_out = open(pickle_name, "wb")
    p = pickle.Pickler(pickle_out)
    p.fast = True
    p.dump(data)

def load_from_pickle(pickle_name):
    file = open(pickle_name, 'rb')
    data = pickle.load(file)
    file.close()
    return data

if __name__ == '__main__':
    timestamp('Preparing dataset')
    dfVocabulary = load_dataset(config.VOCABULARY_DATA_PATH)
    dfClassifier = load_dataset(config.CLASSIFIER_DATA_PATH)

    dfVocabulary = clean_dataset(dfVocabulary)
    dfClassifier = clean_dataset(dfClassifier)

    # Apply the mapping to the data
    def relabel_spec_ids(spec_ids):
        new_spec_ids = []
        for currY in spec_ids:
            if currY in config.MAPPING:
                new_spec_ids.append(config.MAPPING[currY])
            else:
                new_spec_ids.append(0)
        return new_spec_ids


    dfClassifier['newSpecId'] = relabel_spec_ids(dfClassifier.spec_id)
    # Make sure that all classes have several data points to train on
    timestamp(f'Mapped specializations distribution: {Counter(dfClassifier.newSpecId)}')

    timestamp('Training vectorizer')
    vectorizer = TfidfVectorizer()
    vectorizer.fit(dfVocabulary.question)

    timestamp('Embedding questions for classifier')
    embeddedQuestions = vectorizer.transform(dfClassifier.question)

    timestamp('Computing class weights')
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(dfClassifier.newSpecId),
                                                      y=dfClassifier.newSpecId)
    class_weights = dict(enumerate(class_weights))

    timestamp('Training classifier')
    classifier = LogisticRegression(class_weight=class_weights,
                                    random_state=config.RANDOM_STATE)
    classifier.fit(embeddedQuestions, dfClassifier.newSpecId)

    timestamp('Saving models')
    save_to_pickle(vectorizer, config.PICKLE_VECTORIZER_NAME)
    save_to_pickle(classifier, config.PICKLE_CLASSIFIER_NAME)

    timestamp('Finished')
