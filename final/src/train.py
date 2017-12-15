import pickle
import numpy as np
from tqdm import tqdm

with open('data/train.pkl', 'rb') as f:
    train_data = pickle.load(f)

p_minlen = 1000     # 121
p_maxlen = 0        # 603
q_minlen = 1000     # 4
q_maxlen = 0        # 36
for data in train_data:
    p_maxlen = max(p_maxlen, len(data['context']))
    p_minlen = min(p_minlen, len(data['context']))
    q_maxlen = max(q_maxlen, len(data['question']))
    q_minlen = min(q_minlen, len(data['question']))

P = []
Q = []
ans_st = []
ans_ed = []
for i, data in tqdm(enumerate(train_data)):
    p = data['context']
    q = data['question']
    P.append(([np.zeros((300,))] * (p_maxlen - len(p))) + p)
    Q.append(([np.zeros((300,))] * (q_maxlen - len(q))) + q)
    p_offset = data['context_offset']
    ans_st_ = data['ans_st']
    ans_ed_ = data['ans_ed']
    offsum = 0
    for j, off in enumerate(p_offset):
        if ans_st_ <= offsum+off:
            ans_st.append(j + (p_maxlen - len(p)))
            break
        offsum += off
    offsum = 0
    for j, off in enumerate(p_offset):
        offsum += off
        if ans_ed_ <= offsum:
            ans_ed.append(min(p_maxlen, j + 1 + (p_maxlen - len(p))))
            break

del train_data
P = np.array(P)
Q = np.array(Q)
ans_st = np.array(ans_st)
ans_ed = np.array(ans_ed)

from model import *
model = QA(p_maxlen, q_maxlen)
model.compile(loss='mse', optimizer='adam')
model.summary()

model.fit([P, Q], [ans_st, ans_ed], batch_size=128, epochs=30, validation_split=.1, verbose=1)