#############
# Training
#############
PICKLE_VECTORIZER_NAME = 'model_vectorizer.pickle'
PICKLE_CLASSIFIER_NAME = 'model_classifier.pickle'
VOCABULARY_DATA_PATH = 'data_vectorizer.csv'
CLASSIFIER_DATA_PATH = 'data_classifier.csv'
RANDOM_STATE = 42

# This vocabulary defines the mapping of specializations
# If you wish to add a new class (e.g. map spec_id 60 to 5), just add it to the mapping
# Make sure the values make a SEQUENCE
#   if you skip a number (e.g. 1,2,3,5,6), the classifier will fail because 4 is missing and the sequence is incomplete.
MAPPING = {
    52: 4,
    35: 4,
    114: 4,
    115: 4,
    7: 3,
    9: 3,
    8: 3,
    30: 2,
    3: 1,
    46: 1,
    11: 1,
    # 0 is reserved for unmapped (default) classes
}

#############
# Deployment
#############
HOST = 'localhost'
PORT = 5000

#############
# REST API
#############
API_PREFIX = '/api'
API_VERSION = 'v1'
URI_BASE = f'{API_PREFIX}/{API_VERSION}'


