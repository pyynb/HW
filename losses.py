import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class pyyTest(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, sz_batch=128):
        torch.nn.Module.__init__(self)
        self.K = 10
        self.sz_batch = sz_batch
        # Proxy Anchor Initialization
        # 生成class行，embed列的矩阵。共有class个proxy。
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes * self.K, sz_embed).cuda())
        self.pNum = torch.zeros(self.K, nb_classes).bool().cuda()
        for i in range(0, nb_classes):
            self.pNum[0, i] = 1
            # *sz_batch的矩阵
        self.pTotal = torch.ones(nb_classes).int().cuda()
        # print(self.proxies.shape)
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        """
        2020/12/30：
        1.如果不对中心点数量加以限制，容易out of memory。
            solution： 1.限定每个类的最大中心点数量  2.使用不规则的矩阵？
        solved: solution1.
        2.前一个Epoch中增加的中心点，在下一个Epoch中是否继续使用？
            如果使用，则报错：a view of a leaf Variable that requires grad is being used in an in-place operation.
            如果不使用：正在测试。
        solved:1.change self.proxies.clone() to self.proxies.detach();
                2.replace Proxies behind loss.backward() in train.py.
        3.符合条件的input过多。
            目前使用-α（s(x,p)-margin）来界定加点的条件。e.g.  0.1<-α（s(x,p)-margin）<1
            但由于符合条件的点太多，容易内存爆炸和过拟合。
        4.create a new loss.
        """
        # 100=nb_classes（类别数）,T.shape = sz_batch（样本数）,P.shape=[nb_classes,512]
        P = self.proxies.detach()
        pn = self.pNum
        # 计算X和所有Proxies的相似度,cos.shape = batch*(nb_classes*K)
        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        # torch.nonzero temp = cos.reshape(-1, self.nb_classes, self.K)
        # cosSim = torch.zeros(sz_batch, dtype=float).cuda()
        # for i in range(0, sz_batch):
        #    cosSim[i] = cos[:, i:self.nb_classes * self.pNum[i] + 1:self.nb_classes].max().cuda()
        #    # print(cosSim[i])
        #    if cosSim[i] < 0.2:
        #        self.pNum[i] = 1

        # Loss = F.cross_entropy((cosSim, T))
        # cosSim = cos[]
        # Positive和Negative的One Hot向量
        # size=sz_batch*nb_classes,每个样本的
        # if cos.max()<0.2:
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes).cuda()# positive inputs' one hot
        N_one_hot = 1 - P_one_hot
        # -α（s(x,p)-margin）
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg)).cuda()
        neg_exp = torch.exp(self.alpha * (cos + self.mrg)).cuda()
        '''12.17 18：18
        try to reformat the Matrix to 128*10*100
        '''
        # 128*10*100
        pos_exp = pos_exp.reshape(-1, self.K, self.nb_classes)
        neg_exp = neg_exp.reshape(-1, self.K, self.nb_classes)
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # number of positive proxies
        # size=nb_classes
        P_sim = torch.zeros(self.sz_batch, self.nb_classes).cuda()

        '''12/15 21:13
        Located problem:
        P_sim_sum is smaller than it supposed to be
        and N_sim_sum is bigger.'''
        '''12/18 15:11
        can I add Proxies in this loop?
        '''

        for i in range(0, self.sz_batch):
            label = P_one_hot[i].argmax()  # class label of the point
            temp = torch.where(pn, pos_exp[i], torch.zeros_like(pos_exp[i])).cuda()
            P_sim[i] = temp.sum(dim=0).cuda()
            if 0.99 < (temp[:, label].max()) < 1.01:  # condition of add a proxy
                if self.pTotal[label] < 10:
                    index = self.pTotal[label]
                    self.pNum[index, label] = 1  # pNum+=1
                    P[label + self.nb_classes * self.pTotal[label]] = X[i]  # add Proxy
                    self.pTotal[label] = self.pTotal[label] + 1  # pTotal +=1
            label + self.nb_classes * self.pTotal[label]
        P_sim_sum = torch.where(P_one_hot == 1, P_sim, torch.zeros_like(P_sim)).sum(dim=0).cuda()

        N_sim = torch.zeros(self.sz_batch, self.nb_classes).cuda()
        for i in range(0, self.sz_batch):
            N_sim[i] = torch.where(self.pNum, neg_exp[i], torch.zeros_like(neg_exp[i])).sum(
                dim=0).cuda()
        N_sim_sum = torch.where(N_one_hot == 1, N_sim, torch.zeros_like(N_sim)).sum(dim=0).cuda()

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term
        return [loss, P]

    def change_Proxy(self, proxies):
        self.proxies.data = proxies


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        self.K = 10
        # Proxy Anchor Initialization
        # 生成class行，embed列的矩阵。共有class个proxy。
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes * self.K, sz_embed).cuda())
        self.pNum = torch.ones(nb_classes, self.K).bool().cuda()  # class*sz_batch的矩阵
        self.pTotal = torch.ones(nb_classes).int().cuda()
        # print(self.proxies.shape)
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        # nbclasses=nb_classes（类别数）,T.shape = sz_batch（样本数）,P.shape=[nb_classes,512]
        P = self.proxies

        # 计算X和所有Proxies的相似度,cos.shape = batch*(nb_classes*K)
        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        # temp = cos.reshape(-1, self.nb_classes, self.K)
        # cosSim = torch.zeros(sz_batch, dtype=float).cuda()
        # for i in range(0, sz_batch):
        #    cosSim[i] = cos[:, i:self.nb_classes * self.pNum[i] + 1:self.nb_classes].max().cuda()
        #    # print(cosSim[i])
        #    if cosSim[i] < 0.2:
        #        self.pNum[i] = 1

        # print(cosSim.shape)
        # Loss = F.cross_entropy((cosSim, T))
        # cosSim = cos[]
        # Positive和Negative的One Hot向量
        # size=sz_batch*nb_classes,每个样本的
        # if cos.max()<0.2:
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes).cuda()
        # print(P_one_hot.shape)
        N_one_hot = 1 - P_one_hot
        # -α（s(x,p)-margin）
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg)).cuda()
        neg_exp = torch.exp(self.alpha * (cos + self.mrg)).cuda()
        pos_exp = pos_exp.reshape(-1, self.nb_classes, self.K)
        neg_exp = neg_exp.reshape(-1, self.nb_classes, self.K)
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies
        # size=nb_classes

        P_sim = torch.Tensor(128, nb_classes).cuda()
        for i in range(0, 128):
            P_sim[i] = torch.where(self.pNum == 1, pos_exp[i], torch.zeros_like(pos_exp[i])).sum(
                dim=1).cuda()
        P_sim_sum = torch.where(P_one_hot == 1, P_sim, torch.zeros_like(P_sim)).sum(dim=0).cuda()
        N_sim = torch.Tensor(128, nb_classes).cuda()
        for i in range(0, 128):
            N_sim[i] = torch.where(self.pNum == 1, neg_exp[i], torch.zeros_like(neg_exp[i])).sum(
                dim=1).cuda()
        N_sim_sum = torch.where(N_one_hot == 1, N_sim, torch.zeros_like(N_sim)).sum(dim=0).cuda()

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss


# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes=self.nb_classes, embedding_size=self.sz_embed,
                                             softmax_scale=self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50

        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets='semihard')
        self.loss_func = losses.TripletMarginLoss(margin=self.margin)

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings=False)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
