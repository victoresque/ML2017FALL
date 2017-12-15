import jieba
from gensim.models import Word2Vec
import numpy as np
from scipy import spatial

test_file = 'data/testing_data.csv'
w2v_model = Word2Vec.load('data/w2v.mdl')

with open(test_file, 'r', encoding='utf-8') as f:
    lines = list(f)

del lines[0]
questions = []
options = []
for i, line in enumerate(lines):
    _, question_i, options_i = line.split(',')
    options_i = options_i.split(':')[1:]
    questions.append(question_i.replace('A:', '').replace('B:', '').replace('C:', '').replace('D:', ''))
    options_i = [opt.replace('\t','').replace('\n','').replace('A', '').replace('B', '').replace('C', '').replace('D', '')
        for opt in options_i]
    options.append(options_i)

print(questions[:2])
print(options[:2])

q_vec = []
for q in questions:
    seg_q = list(jieba.cut(q))
    q_vec_list = []
    for w in seg_q:
        if w in w2v_model:
            q_vec_list.append(w2v_model[w])
    q_vec.append(q_vec_list)

opt_vec = []
for opts_1Q in options:
    opt_vec_1Q = []
    for opt in opts_1Q:
        seg_opt = list(jieba.cut(opt))
        opt_vec_list = []
        for w in seg_opt:
            if w in w2v_model:
                opt_vec_list.append(w2v_model[w])
        opt_vec_1Q.append(opt_vec_list)
    opt_vec.append(opt_vec_1Q)

q_vec_mean = []
for vec_list in q_vec:
    vec_list = np.array(vec_list)
    q_vec_mean.append(np.mean(vec_list,axis=0))

#q_vec_mean = np.array(q_vec_mean)

opt_vec_mean = []
for i, opts in enumerate(opt_vec):
    means_for_options = []
    for opt in opts:
        opt = np.array(opt)
        means_for_options.append(np.mean(opt,axis=0))
    opt_vec_mean.append(means_for_options)
#opt_vec_mean = np.array(opt_vec_mean)

ans = []
for i, qmean in enumerate(q_vec_mean):
    sim_6 = []
    for opt in opt_vec_mean[i]:
        sim_1 = 1-spatial.distance.cosine(qmean, opt)
        sim_6.append(sim_1)
    ans.append(np.argmax(sim_6))

with open('ans.csv', 'w') as f:
    f.write('id,ans\n')
    for i, a in enumerate(ans):
        f.write(str(i+1)+','+str(a)+'\n')
print(len(ans))
print(ans[:10])
