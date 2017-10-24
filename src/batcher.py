import numpy as np
from sklearn.externals import joblib
import random

class Batcher:
    def __init__(self,storage,data,batch_size,context_length,id2vec):
        self.context_length = context_length
        self.storage = storage
        self.data = data
        self.num_of_samples = int(data.shape[0])
        self.dim = 300 #len(id2vec[0])
        self.num_of_labels = data.shape[1] - 4  - 70 
        self.batch_size = batch_size
        self.batch_num = 0
        self.max_batch_num = int(self.num_of_samples / self.batch_size)

        self.id2vec = id2vec
        self.pad  = np.zeros(self.dim)
        self.pad[0] = 1.0
        self.bags = {}
        self.bags_id = []
        self.label_divide_data = {}

    def create_input_output(self,row):
        s_start = row[0]
        s_end = row[1]
        e_start = row[2]
        e_end = row[3]
        labels = row[74:]
        features = row[4:74]
        seq_context = np.zeros((self.context_length*2 + 1,self.dim))        
        temp = [ self.id2vec[self.storage[i]][:self.dim] for i in range(e_start,e_end)]
        mean_target = np.mean(temp,axis=0)
        
        j = max(0,self.context_length - (e_start - s_start))
        for i in range(max(s_start,e_start - self.context_length),e_start):
            seq_context[j,:] = self.id2vec[self.storage[i]][:self.dim]
            j += 1
        seq_context[j,:] = np.ones_like(self.pad)
        j += 1
        for i in range(e_end,min(e_end+self.context_length,s_end)):
            seq_context[j,:] = self.id2vec[self.storage[i]][:self.dim]
            j += 1

        return seq_context, mean_target, labels, features
        

    def next(self):
        X_context = np.zeros((self.batch_size,self.context_length*2+1,self.dim))
        X_target_mean = np.zeros((self.batch_size,self.dim)) 
        Y = np.zeros((self.batch_size,self.num_of_labels))
        F = np.zeros((self.batch_size,70),np.int32)
        for i in range(self.batch_size):
            X_context[i,:,:], X_target_mean[i,:], Y[i,:], F[i,:] = self.create_input_output(self.data[self.batch_num * self.batch_size + i,:])
        self.batch_num = (self.batch_num + 1) % self.max_batch_num
        return [X_context, X_target_mean, Y, F]
                                        
    def shuffle(self):
        np.random.shuffle(self.data)


    def get_id2level(self,target_dim,id2label):
        level_set = {}
        for x in range(0,target_dim):
            level=len(id2label[x].split('/')) - 2
            level_set[x] = level
        return level_set

    def divide_data(self,target_dim):
        
        for j in range(target_dim):
            self.label_divide_data[j] = []

        for i in range(len(self.data)):
            row = self.data[i,:]
            label = row[74:]
            for j in range(len(label)):
                if(label[j]==1):
                    self.label_divide_data[j].append(i)

        for k in self.label_divide_data:
            print str(k)+":"+str(len(self.label_divide_data[k]))

        return  self.label_divide_data


    # def create_bags(self,id2word):
    #     for i in range(len(self.data)):
    #         row = self.data[i,:]
    #         e_start = row[2]
    #         e_end = row[3]
    #         label = row[74:]
    #         label_tag = [l for l in range(len(label)) if label[l]==1]
    #         # entity = str([id2word[self.storage[j]] for j in range(e_start,e_end)])
    #         entity = str([id2word[self.storage[j]] for j in range(e_start,e_end)])+str(label_tag)
    #         if entity not in self.bags:
    #             self.bags[entity] = [i]
    #         else:
    #             self.bags[entity].append(i)

    #     print "bag nums:"+str(len(self.bags))

    #     bags_error={}
    #     for key in self.bags:
    #         # print key
    #         self.bags_id.append(key)
    #         first = self.bags[key][0]
    #         row = self.data[first,:]
    #         first_label = row[74:]
    #         # print len(self.bags[key])
    #         # print self.bags[key]
    #         for i in self.bags[key]:
    #             # row = self.data[i,:]
    #             # if i == first:
    #             #     continue
    #             # print "i!=first"
    #             label_tmp = self.data[i,74:]
    #             # assert (list(label_tmp)==list(first_label)), "error:"+str(key)
    #             if not(list(label_tmp)==list(first_label)):
    #                 bags_error[key]=1
    #     self.max_bag_batch_num = int(len(self.bags) / self.batch_size)
    #     print "max_bag_batch_num:"+str(self.max_bag_batch_num)
    #     print "error num:"+str(len(bags_error))
    #     # print  bags_error

    # def next_bag_batch(self):
        
    #     len_of_each_bag = []
    #     for i in range(self.batch_size):
    #         bag_name = self.bags_id[self.batch_num * self.batch_size + i]
    #         bag = self.bags[bag_name]
    #         len_of_each_bag.append(len(bag))

    #     total_len = sum(len_of_each_bag)

    #     X_context = np.zeros((total_len,self.context_length*2+1,self.dim))
    #     X_target_mean = np.zeros((total_len,self.dim)) 
    #     Y = np.zeros((total_len,self.num_of_labels))
    #     F = np.zeros((total_len,70),np.int32)

    #     sum_len = 0
    #     for i in range(self.batch_size):
    #         bag_name = self.bags_id[self.batch_num * self.batch_size + i]
    #         bag = self.bags[bag_name]
    #         for j in range(len(bag)):
    #             instance = bag[j]
    #             X_context[sum_len+j,:,:], X_target_mean[sum_len+j,:], Y[sum_len+j,:], F[sum_len+j,:] = self.create_input_output(self.data[instance,:])
    #         sum_len += len(bag)

    #     assert sum_len==total_len,"sum_len!=total_len !"
    #     self.batch_num = (self.batch_num + 1) % self.max_bag_batch_num
    #     # print Y
    #     return [X_context, X_target_mean, Y, F, len_of_each_bag]


    def create_bags(self,id2word,bag_size):
        for i in range(len(self.data)):
            row = self.data[i,:]
            e_start = row[2]
            e_end = row[3]
            label = row[74:]
            label_tag = [l for l in range(len(label)) if label[l]==1]
            entity = str([id2word[self.storage[j]] for j in range(e_start,e_end)])+str(label_tag)
            # entity = str(label_tag)
            if entity not in self.bags:
                self.bags[entity] = [[i]]
            else:
                if len(self.bags[entity][-1]) < bag_size:
                    self.bags[entity][-1].append(i)
                else:
                    self.bags[entity].append([i])
        bags_error={}
        avg_bag_size=0.0
        for key in self.bags:
            # print key
            # print str(key) +" bag num:"+str(len(self.bags[key]))
            for n in range(len(self.bags[key])):
                self.bags_id.append((key,n))
                # print str(n)+" bag_size:"+str(len(self.bags[key][n]))
                avg_bag_size += len(self.bags[key][n])
        print "total bag nums:"+str(len(self.bags_id))
        print "avg bag size:"+str(avg_bag_size/float(len(self.bags_id)))
        self.max_bag_batch_num = int(len(self.bags_id) / self.batch_size)
        
        return len(self.bags_id)

    def next_bag_batch(self):
        len_of_each_bag = []

        for i in range(self.batch_size):
            bag_set_id = self.bags_id[self.batch_num * self.batch_size + i]
            bag_set = bag_set_id[0]
            bag_id = bag_set_id[1]
            bag = self.bags[bag_set][bag_id]
            len_of_each_bag.append(len(bag))
        total_len = sum(len_of_each_bag)

        X_context = np.zeros((total_len,self.context_length*2+1,self.dim))
        X_target_mean = np.zeros((total_len,self.dim)) 
        Y = np.zeros((total_len,self.num_of_labels))
        F = np.zeros((total_len,70),np.int32)

        sum_len = 0
        for i in range(self.batch_size):
            bag_set_id = self.bags_id[self.batch_num * self.batch_size + i]
            bag_set = bag_set_id[0]
            bag_id = bag_set_id[1]
            bag = self.bags[bag_set][bag_id]
            for j in range(len(bag)):
                instance = bag[j]
                X_context[sum_len+j,:,:], X_target_mean[sum_len+j,:], Y[sum_len+j,:], F[sum_len+j,:] = self.create_input_output(self.data[instance,:])
            sum_len += len(bag)

        assert sum_len==total_len,"sum_len!=total_len !"
        self.batch_num = (self.batch_num + 1) % self.max_bag_batch_num
        # print Y
        return [X_context, X_target_mean, Y, F, len_of_each_bag]



    def shuffle_bags(self):
        np.random.shuffle(self.bags_id)

    def get_label_hierarchy(self,id2label,target_dim):
        level_set = [[],[],[]]
        label_hierarchy = []
        for x in range(0,target_dim):
            tmp = {'father':[],'son':[],'sibling':[]}
            level_x=len(id2label[x].split('/'))
            for y in range(0,target_dim):
                if y==x:
                    continue
                else:
                    level_y=len(id2label[y].split('/'))
                    if id2label[y] in id2label[x] and level_x-level_y==1:
                        tmp['father'].append(y)
                    if id2label[x] in id2label[y] and level_y-level_x==1:
                        tmp['son'].append(y)
                    
                    # if level_x==level_y:
                    #   tmp['sibling'].append(y)
            tmp['level'] = level_x-2
            label_hierarchy.append(tmp)
            # print level_x
            level_set[level_x-2].append(x)

        return label_hierarchy


    # def label2path(self,id2label,target_dim,label2id,target):
    #     # level_set = [[],[],[]]
    #     # label_hierarchy = []
    #     # label2path = {}
    #     # for x in range(0,target_dim):
    #     #     nodes = id2label[x].split('/')
    #     #     label2path[x]=[]
    #     #     path=""
    #     #     for n in nodes:
    #     #         path = path+"/"+n
    #     #         label2path.append(label2id[path])
    #     path_target = []
    #     for i in range(target_dim)

    #     return label2path

    def get_path(self,node,bag_target,ancestors,label_hierarchy):
        son=[]
        tmp=[]
        for x in ancestors:
            tmp.append(x)
        tmp.append(node)
        for j in label_hierarchy[node]['son']:
            if bag_target[j]==1:
                son.append(j)
        if son == []:
             self.path.append(tmp)
        else:
            for s in son:
                self.get_path(s,bag_target,tmp,label_hierarchy)

    def transform2path(self,label_hierarchy,target_dim):
        for i in range(len(self.data)):
            label = self.data[i,74:]
            self.path = []
            for j in range(4):
                if label[j]==1:
                    ancestors=[]
                    self.get_path(j,label,ancestors,label_hierarchy)
            path_label = np.zeros(target_dim)
            for p in self.path:
                endnode = p[-1]
                path_label[endnode] = 1
            # print "label:"+str(label)
            # print "path_label:"+str(path_label)
            self.data[i,74:] = path_label

    def statistic(self,target_dim,id2label):
        entity_set = {}
        id2level = self.get_id2level(target_dim,id2label)

        levels = {}
        for i in range(len(self.data)):
            row = self.data[i,:]
            e_start = row[2]
            e_end = row[3]
            entity = str([self.storage[j] for j in range(e_start,e_end)])
            if entity not in entity_set:
                entity_set[entity] = 1
            else:
                entity_set[entity] += 1

            label = row[74:]
            level = []
            for j in range(len(label)):
                if(label[j]==1):
                    level.append(id2level[j])
            if max(level) not in levels:
                levels[max(level)] = 1
            else:
                levels[max(level)] += 1

        for x in levels:
            print "level "+str(x)+" num:"+str(levels[x])+" percent:"+str(float(levels[x])/len(self.data))

        ent_freq_set = {}
        total_freq = 0
        max_tmp = 0
        for i in entity_set:
            total_freq += entity_set[i]
            if entity_set[i] not in ent_freq_set:
                ent_freq_set[entity_set[i]] = 1
            else:
                ent_freq_set[entity_set[i]] += 1

        print "entity num:"+str(len(entity_set))
        print "average entity frequency(1 entity appears in many sentence):" + str(float(total_freq)/len(entity_set))
        # print ent_freq_set
        
        return entity_set


def filter(context_data, mention_representation_data, target_data, feature_data, scores):
    num_bag = 2
    filter_context_data = []
    filter_mention_representation_data = []
    filter_target_data = []
    filter_feature_data = []
    filter_scores = []
    for i in range(0,len(context_data)/num_bag,num_bag):
        pro = []
        for n in range(num_bag):
            pro.append(1)
            for j in range(len(target_data[0])):
                if target_data[i+n,j] == 1:
                    pro[n] *= scores[i+n,j]

        tid,pro = max(enumerate(pro),key=lambda x: x[1])
        filter_context_data.append(context_data[i+tid])
        filter_mention_representation_data.append(mention_representation_data[i+tid])
        filter_target_data.append(target_data[i+tid])
        filter_feature_data.append(feature_data[i+tid])

    filter_context_data = np.array(filter_context_data)
    filter_mention_representation_data = np.array(filter_mention_representation_data)
    filter_target_data = np.array(filter_target_data)
    filter_feature_data = np.array(filter_feature_data)
    filter_scores = np.array(filter_scores)

    return [filter_context_data, filter_mention_representation_data, filter_target_data, filter_feature_data]