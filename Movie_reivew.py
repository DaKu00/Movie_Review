import pandas as pd

train_data = pd.read_json('train_data.json')
test_data = pd.read_json('test_set.json')
# from konlpy.tag import Okt
# okt = Okt()
# # train_data = pd.read_csv('ratings_train.txt', '\t')
# train_data.dropna(inplace = True)
# def tokenize(doc):
#     #형태소와 품사를 join
#     return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]
# train_data = [(tokenize(row[1]), row[2]) for row in train_data.values]
train_data.columns = ['review', 'ratings']
test_data.columns = ['review', 'ratings']
test_data.dropna(inplace = True)
train_data.dropna(inplace = True)
train_data

train_data = list(zip(train_data['review'], train_data['ratings']))
test_data = list(zip(test_data['review'], test_data['ratings']))
tokens = [t for d in train_data for t in d[0]]
print("토큰개수:", len(tokens))

import nltk
text = nltk.Text(tokens, name='NMSC')

#토큰개수
print(len(text.tokens))

#중복을 제외한 토큰개수
print(len(set(text.tokens)))

#출력빈도가 높은 상위 토큰 10개
print(text.vocab().most_common(10))

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.figure(figsize=(20,10))
plt.rc('font', family = fm.FontProperties(fname = 'C:/Windows/Fonts/malgun.ttf').get_name())
text.plot(50)

import numpy as np
FREQUENCY_COUNT = 100;
selected_words = [f[0] for f in text.vocab().most_common(FREQUENCY_COUNT)]

# 단어리스트에서 상위 100개중 포함되는 단어들의 갯수 리스트
def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

# train_data에서 정답을 제외한 문장을 각 단어별로 갯수리스트의 리스트로 만듦
x_train = [term_frequency(d) for d,_ in train_data]
x_test = [term_frequency(d) for d,_ in test_data]

# 1 또는 0으로 긍정부정의 정답을 리스트로 만듦
y_train = [c for _,c in train_data]
y_test = [c for _,c in test_data]
x_train = np.asarray(x_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')

y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')
from keras.models import Sequential
from keras.layers import Dense, Activation


model = Sequential()
model.add(Dense(64, activation='relu', input_dim = FREQUENCY_COUNT))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=512)

results = model.evaluate(x_test, y_test)

review = "아주 재미 있어요"
def predict_review(review):
    token = tokenize(review)
    tfq = term_frequency(token)
    data = np.expand_dims(np.asarray(tfq).astype('float32'), axis=0)
    score = float(model.predict(data))
    if(score > 0.5):
        print(f"{review} ==> 긍정 ({round(score*100)}%)")
    else:
        print(f"{review} ==> 부정 ({round((1-score)*100)}%)")
predict_review("아주 재미 없어요")

