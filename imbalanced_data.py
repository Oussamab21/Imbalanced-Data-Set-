# Renner and Bouldjedri
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
from random import shuffle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix


# computes accuracy, precision, recall from confusion matrix
def acc_prec_rec(confusion):
    acc = float(confusion[0][0] + confusion[1][1]) / float(confusion[0][0] + confusion[1][1]
                                                           + confusion[1][0] + confusion[0][1])
    if confusion[0][1] > 0: # make sure there are predicted positive classes (so not / by zero error)
        prec = float(confusion[1][1]) / float(confusion[1][1] + confusion[0][1])
    else:
        prec = 0.0
    rec = float(confusion[1][1]) / float(confusion[1][1] + confusion[1][0])
    print 'Accuracy: {}'.format(acc)
    print 'Precision: {}'.format(prec)
    print 'Recall: {}'.format(rec)

def load_train_data():
    train_feat = np.loadtxt("data/fraud_train_small.csv", delimiter=',')
    train_labels = np.loadtxt("data/fraud_train_label_small.csv")
    return train_feat, train_labels

## load and create validation set
train_feat, train_labels = load_train_data()
train = list(zip(train_feat,train_labels))
shuffle(train)
validation = train[0:10000]
train = train[10000:]
val_feats, val_labels = zip(*validation)
train_feat, train_labels = zip(*train)
print 'Validation dataset shape {}'.format(Counter(val_labels))

## oversampling
maj_examples = [x for x in train if x[1] == 0.0][0:10000]  # under sample to make dataset small enough to run on my computer
min_examples = [x for x in train if x[1] == 1.0]
train = maj_examples + min_examples
shuffle(train)
smaller_features, smaller_labels = zip(*train)
print 'Original dataset shape {}'.format(Counter(train_labels))

# SMOTE
sm = SMOTE(random_state=1)
tr_feat_smote, tr_labels_smote = sm.fit_sample(smaller_features, smaller_labels)
print 'New dataset shape {}'.format(Counter(tr_labels_smote))



# Random Oversampling
os = RandomOverSampler(random_state=1)
tr_feat_ros, tr_labels_ros = os.fit_sample(smaller_features, smaller_labels)
print 'New dataset shape {}'.format(Counter(tr_labels_ros))

clf = AdaBoostClassifier(n_estimators=50)
clf.fit(tr_feat_ros, tr_labels_ros)
pred = clf.predict(val_feats)
con = confusion_matrix(val_labels, pred)
acc_prec_rec(con)

exit(0)
# load test data
test = np.loadtxt("data/fraud_test_small.csv", delimiter=',')
pred = clf.predict(test)
f = open("RanOVeroutput.txt", 'w')
for p in pred:
    f.write(str(p) + "\n")
f.close()


