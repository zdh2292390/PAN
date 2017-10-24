# -*- coding: utf-8 -*-
import argparse
from sklearn.externals import joblib
from src.model.nn_torch import Model
from src.batcher import Batcher,filter
from src.hook import acc_hook, save_predictions
import datetime
import numpy as np
import torch.optim as optim
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("dataset",help="dataset to train model",choices=["figer","gillick"])
parser.add_argument("encoder",help="context encoder to use in model",choices=["averaging","lstm","attentive","type_att"])
parser.add_argument('--feature', dest='feature', action='store_true')
parser.add_argument('--no-feature', dest='feature', action='store_false')
parser.set_defaults(feature=False)
parser.add_argument('--hier', dest='hier', action='store_true')
parser.add_argument('--no-hier', dest='hier', action='store_false')
parser.set_defaults(hier=False)
parser.add_argument('--gaussian', dest='gaussian', action='store_true')
parser.set_defaults(gaussian=False)
parser.add_argument('--regularize', dest='regularize', action='store_true')
parser.set_defaults(regularize=False)
parser.add_argument('--trace', dest='trace', action='store_true')
parser.set_defaults(trace=False)
parser.add_argument('--bags', dest='bags', action='store_true')
parser.set_defaults(bags=False)
parser.add_argument('--path', dest='path', action='store_true')
parser.set_defaults(path=False)
parser.add_argument("bag_strategy",help="strategy to reduce bag",choices=["one","att","tatt","patt"])
parser.add_argument("-batch_size", action="store", dest="batch_size", type=int)
parser.add_argument("-bag_size", action="store", dest="bag_size", type=int)
parser.add_argument("-resultpath", action="store", dest="resultpath")
args = parser.parse_args()

print "encoder:"+str(args.encoder)
print "feature:"+str(args.feature)+" hier:"+str(args.hier)
print "batch_size:"+str(args.batch_size)
print "bags:"+str(args.bags)
print "bag_size:"+str(args.bag_size)
print "bag_strategy:"+str(args.bag_strategy)
print "resultpath:"+str(args.resultpath)
print "path:"+str(args.path)

print "Creating the model"
starttime=datetime.datetime.now()
if args.gaussian == True:
    model = Model(type=args.dataset,encoder=args.encoder,hier=args.hier,feature=args.feature, gaussian=args.gaussian, margin=args.margin, negtive_size=args.negtive_size, gaussian_dim=args.gaussian_dim, regularize=args.regularize, minval=args.min, maxval=args.max)
else:
    model = Model(type=args.dataset,encoder=args.encoder,hier=args.hier,feature=args.feature, bags=args.bags, bag_strategy=args.bag_strategy)
endtime=datetime.datetime.now()
print "initiate time "+str((endtime-starttime).total_seconds())

print "Loading the dictionaries"
d = "Wiki" if args.dataset == "figer" else "OntoNotes"
target_dim = 113 if args.dataset == "figer" else 89
dicts = joblib.load("data/"+d+"/dicts_"+args.dataset+".pkl")
# print dicts["id2label"]

model.coarse_type_set(dicts["id2label"])
print "coarse_set"
print model.coarse_set

print "Loading the datasets"
train_dataset = joblib.load("data/"+d+"/train_"+args.dataset+".pkl")
dev_dataset = joblib.load("data/"+d+"/dev_"+args.dataset+".pkl")
test_dataset = joblib.load("data/"+d+"/test_"+args.dataset+".pkl")

print "train_size:", train_dataset["data"].shape[0]
print "dev_size: ", dev_dataset["data"].shape[0]
print "test_size: ", test_dataset["data"].shape[0]

print "Creating batchers"
# batch_size : 1000, context_length : 10
batch_size = args.batch_size
train_batcher = Batcher(train_dataset["storage"],train_dataset["data"],batch_size,10,dicts["id2vec"])
dev_batcher = Batcher(dev_dataset["storage"],dev_dataset["data"],dev_dataset["data"].shape[0],10,dicts["id2vec"])
test_batcher = Batcher(test_dataset["storage"],test_dataset["data"],test_dataset["data"].shape[0],10,dicts["id2vec"])

# print "divide data to labels"
# train_batcher.divide_data(target_dim)
label_hierarchy = train_batcher.get_label_hierarchy(dicts['id2label'],target_dim)
if args.path:
    print "transform label to path"
    train_batcher.transform2path(label_hierarchy, target_dim)
    dev_batcher.transform2path(label_hierarchy, target_dim)
    test_batcher.transform2path(label_hierarchy, target_dim)
if args.bags:
    print "Creating bags"
    bag_num = train_batcher.create_bags(dicts['id2word'],args.bag_size)
    step_par_epoch = bag_num/batch_size
    # step_par_epoch = 629313/batch_size if args.dataset == "figer" else 24419/batch_size
else:
    step_par_epoch = 2000000/batch_size if args.dataset == "figer" else 250000/batch_size

# model = nn.DataParallel(model, device_ids=[0,1]).cuda()

optimizer = optim.Adam(model.parameters())
print "model parameters:"
for name, param in model.named_parameters():
    print name

print "start trainning"
time1 = datetime.datetime.now()
for epoch in range(20):
    train_batcher.shuffle()
    train_batcher.shuffle_bags()

    print "epoch",epoch
    time2 = datetime.datetime.now()
    for i in range(step_par_epoch):
    	time3 = datetime.datetime.now()
        model.zero_grad()

        if not args.bags:    
            context_data, mention_representation_data, target_data, feature_data = train_batcher.next()
            loss = model(context_data, mention_representation_data, target_data, feature_data,0.5)
            print "loss:"+str(loss.data)
            loss.backward()
            optimizer.step()
        else:
            context_data, mention_representation_data, target_data, feature_data, len_of_each_bag = train_batcher.next_bag_batch()
            # scores = model.predict(context_data, mention_representation_data, feature_data,0)
            time4 = datetime.datetime.now()
            # loss = model(context_data, mention_representation_data, target_data, feature_data, 0.5, len_of_each_bag,label_hierarchy)
            loss = model(context_data, mention_representation_data, target_data, feature_data, 0.5, len_of_each_bag,label_hierarchy,"mul")
            endtime = datetime.datetime.now()
            print "forward time:"+str((endtime-time4).total_seconds())
            print "loss:"+str(loss.data)
            
            loss.backward(retain_graph=True)
            optimizer.step()
            # for label in range(target_dim):
            #     time4 = datetime.datetime.now()
            #     loss = model(context_data, mention_representation_data, target_data, feature_data, 0.5, scores, len_of_each_bag,label)
            #     print "loss:"+str(loss.data)
            #     loss.backward(retain_graph=True)
            #     optimizer.step()
            #     endtime = datetime.datetime.now()
            #     print str(epoch)+" "+str(i)+" label:"+str(label)+" train time: "+str((endtime-time4).total_seconds())

        endtime = datetime.datetime.now()
        print str(epoch)+" "+str(i)+" train time: "+str((endtime-time3).total_seconds())
        # print "loss:"+str(loss)
    endtime = datetime.datetime.now()
    print "epoch"+str(epoch)+" train time: "+str((endtime-time2).total_seconds())
    
    print "------dev--------"
    context_data, mention_representation_data, target_data, feature_data = dev_batcher.next()
    scores = model.predict(context_data, mention_representation_data, feature_data,0)
    acc_hook(scores, target_data, args.gaussian, 0, 1, args.path, label_hierarchy)
    if args.gaussian:
        np.savetxt(args.resultpath +"/scores_epoch"+str(epoch), scores, fmt ='%f')
        scores = np.sort(a = scores,axis = 1)
        np.savetxt(args.resultpath+"/sorted_scores_epoch"+str(epoch), scores, fmt='%f')
    print "-----test--------"
    context_data, mention_representation_data, target_data, feature_data = test_batcher.next()
    scores = model.predict(context_data, mention_representation_data, feature_data, args.gaussian)
    acc_hook(scores, target_data, args.gaussian, 0, 1, args.path, label_hierarchy)

endtime=datetime.datetime.now()
print "total train time: "+str((endtime-time1).total_seconds())

print "Training completed.  Below are the final test scores: "
print "-----test--------"
context_data, mention_representation_data, target_data, feature_data = test_batcher.next()
scores = model.predict(context_data, mention_representation_data, feature_data,0)
acc_hook(scores, target_data, args.gaussian,args.path,label_hierarchy)
fname = args.dataset + "_" + args.encoder + "_" + str(args.feature) + "_" + str(args.hier) + ".txt"
# fname = args.resultpath + "/prediction"
save_predictions(scores, target_data, dicts["id2label"], fname, args.gaussian)

print "Cheers!"