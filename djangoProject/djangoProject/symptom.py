import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class Symptom:

    def __init__(self):
        training = pd.read_csv("C:\\Users\dylan\Desktop\Year 4 Sem2\djangoProject\datasets\Training.csv")
        symptoms = list(training)
        self.symptoms_list = []
        for s in symptoms:
            symptom = s.replace('_', " ")
            self.symptoms_list.append(symptom)

    def symptom_checker(self, ):
        pass

    def compare_symptoms(self, symptoms_input):
        stop_words = set(stopwords.words('english'))

        symptom_tokens = word_tokenize(symptoms_input)
        print(f"Before removing stop words: {symptom_tokens}")
        symptoms = []

        for w in symptom_tokens:
            if w not in stop_words:
                symptoms.append(w)
        print(f"After removing stop words:{symptoms}")

        confirmed_symptoms = []
        for s in symptoms:
            w1 = self.word2vec(s)
            for sl in self.symptoms_list:
                w2 = self.word2vec(sl)
                if self.cosdis(w1, w2) > 0.8:
                    # print(s, sl, self.cosdis(w1, w2))
                    if sl not in confirmed_symptoms:
                        confirmed_symptoms.append(sl)
                # if jaccard_similarity > 0.6:
                #     print(s, sl, jaccard_similarity)

        print("Matching user symptoms to possible dataset symptoms: ")
        count = 1
        for s in confirmed_symptoms:
            print(f"{count}: {s}")
            count += 1

        updated_symptoms = []
        for s in confirmed_symptoms:
            updated_symptoms.append(s.replace(" ", "_"))
        return updated_symptoms

    def word2vec(self, word):
        from collections import Counter
        from math import sqrt

        # count the characters in word
        cw = Counter(word)
        # precomputes a set of the different characters
        sw = set(cw)
        # precomputes the "length" of the word vector
        lw = sqrt(sum(c * c for c in cw.values()))

        # return a tuple
        return cw, sw, lw

    def cosdis(self, v1, v2):
        # which characters are common to the two words?
        common = v1[1].intersection(v2[1])
        # by definition of cosine distance we have
        return sum(v1[0][ch] * v2[0][ch] for ch in common) / v1[2] / v2[2]

    def get_diag(self, symptoms):
        df = pd.read_csv(
            'C:\\Users\dylan\Desktop\Year 4 Sem2\djangoProject\datasets\Training.csv')
        X = df.iloc[:, :-1]
        y = df['prognosis']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
        rf_clf = DecisionTreeClassifier()
        rf_clf.fit(X_train, y_train)

        symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
        input_vector = np.zeros(len(symptoms_dict))
        for item in symptoms:
            input_vector[[symptoms_dict[item]]] = 1

        diagnosis = rf_clf.predict([input_vector])

        return diagnosis[0]

    def get_description(self, disease):
        data = pd.read_csv('C:\\Users\dylan\Desktop\Year 4 Sem2\djangoProject\datasets\symptom_Description.csv')
        description = data.loc[data['Disease'] == disease, 'Description']
        print(description.values[0])
        return str(description.values[0])

    def get_treatment(self, disease):
        data = pd.read_csv('C:\\Users\dylan\Desktop\Year 4 Sem2\djangoProject\datasets\symptom_precaution.csv')
        treatment = data.loc[data['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()

        return treatment[0]