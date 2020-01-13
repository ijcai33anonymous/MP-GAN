import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


class CLASSIFIER:
    def __init__(self, opt, model, _train_X, _train_Y,_test_seen_X,_test_seen_Y,_test_novel_X, _test_novel_Y, seenclasses,
                 novelclasses, _nclass, weights):
        self.opt = opt
        self.weights = weights
        self.train_X =  _train_X
        self.train_Y = _train_Y

        self.test_seen_feature = _test_seen_X
        self.test_seen_label = _test_seen_Y
        self.test_novel_feature = _test_novel_X
        self.test_novel_label = _test_novel_Y

        self.seenclasses = seenclasses
        self.novelclasses = novelclasses
        self.seenclasses_num = self.seenclasses.shape[0]
        self.novelclasses_num = self.novelclasses.shape[0]

        self.batch_size = self.opt.cls_batch_size
        self.nepoch = self.opt.cls_epoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        print('self.input_dim')
        print(self.input_dim)

        self.average_loss = 0

        self.model = []
        #
        for cls in model:
            self.model.append(cls.cuda())

        self.criterion = nn.NLLLoss()

        self.input = torch.FloatTensor(self.batch_size, self.input_dim).cuda()
        self.label = torch.LongTensor(self.batch_size).cuda()
        self.lr = self.opt.cls_lr

        self.optimizers = []
        for num in range(len(self.model)):
            f = list(filter(lambda x:  x.requires_grad, self.model[num].parameters()))
            self.optimizers.append(optim.Adam(f, lr=self.lr,  betas=(0.9, 0.999)))#

        self.criterion.cuda()
        self.input = self.input.cuda()
        self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        self.loss = 0
        self.current_epoch = 0
        self.acc = 0
        self.acc = self.fit_zsl()

    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):

                for cls in self.model:
                    cls.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = self.input
                labelv = self.label
                for num in range(len(self.model)):
                    output = self.model[num](inputv[:, num*512:(num+1)*512])
                    loss = self.criterion(output, labelv)
                    mean_loss += loss.item()  # data[0]
                    loss.backward()
                    self.optimizers[num].step()

            self.current_epoch += 1

            acc = 0
            with torch.no_grad():
                acc, weighted_acc = self.val(self.test_novel_feature, self.test_novel_label, self.novelclasses)
            print("acc: ", acc, "weighted_acc: ", weighted_acc)

            if best_acc > acc:
                best_acc = acc

        return best_acc


    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]


    def val(self, test_X, test_label, target_classes):
        # weights = [0.999325, 0.62739927, 0.5503919, 0.72866416, 0.41019028, 0.64912724, 0.35950154]
        # weights = [0.7746, 0.5016, 0.4214, 0.0233, 0.4582, 0.0685, 0.5237]
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        weighted_predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            outputs = torch.zeros([end-start, target_classes.size(0)]).cuda()
            weighted_outputs = torch.zeros([end-start, target_classes.size(0)]).cuda()
            for num in range(len(self.model)):
                output = self.model[num](test_X[start:end, num*512:(num+1)*512].cuda())
                outputs += output
                weighted_outputs += self.weights[num] * output
            # output = self.model(test_X[start:end].cuda())
            _, predicted_label[start:end] = torch.max(outputs.data, 1)
            _, weighted_predicted_label[start:end] = torch.max(weighted_outputs.data, 1)

            start = end

        acc = self.compute_per_class_acc(map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        weighted_acc = self.compute_per_class_acc(map_label(test_label, target_classes), weighted_predicted_label, target_classes.size(0))

        return acc, weighted_acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):

        per_class_accuracies = torch.zeros(nclass).float().cuda().detach()

        target_classes = torch.arange(0, nclass, out=torch.LongTensor()).cuda() #changed from 200 to nclass on 24.06.
        predicted_label = predicted_label.cuda()
        test_label = test_label.cuda()

        for i in range(nclass):

            is_class = test_label==target_classes[i]

            per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(), is_class.sum().float())

        return per_class_accuracies.mean()

class TrainDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train_X, train_Y):

        self.train_X = train_X
        self.train_Y = train_Y.long()

    def __len__(self):
        return self.train_X.size(0)

    def __getitem__(self, idx):

        return {'x': self.train_X[idx,:], 'y': self.train_Y[idx] }


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label