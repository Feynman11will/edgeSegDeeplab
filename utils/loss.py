import torch
import torch.nn as nn
import logging
import numpy as np
from skimage import io,data,morphology
import cv2
logging.basicConfig('../logPath/edgeloss.log',level=logging.INFO)


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False,index_list=None):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.index_list = index_list

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode =='edgece':
            return self.EdgeCeloss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()
        
        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def EdgeCeloss(self, logit, target,alpha=0.2, beta=0.6):
        """
        @ edgeCeloss = edgeLoss+ crossEntropy loss
        """
        n, c, h, w = logit.size()
        Edge, Weight = self.splitMask2Edge(target)

        lossSeg = self.CrossEntropyLoss(logit, target)
        lossEdge = 0
        loss = lossSeg + lossEdge

        return loss
        
    def splitMask2Edge(self, target,nc=20):
        '''
        @ 将一个mask 按照分类数量分裂成c个mask边界框
        '''
        n,w,h = target.size()
        logging.info(f'n,w,h :{(n,w,h)}')
        logging.info(f'target type:{type(target)}')
        
        if self.index_list==None:
            self.index_list = [i+1 for i in range(nc)]

        target_list = torch.chunk(target,n, dim=0)
        logging.info(f'target_list len:{len(target_list)}')
        maskEdgeList  = []
        weightEdgeList = []


        for target_ in target_list:
            maskList = []
            weightList = []
            for index in self.index_list:
                IndexTensor = (target_==index).float()
                edge,weight = self.getEdgeWeight(IndexTensor,blurSize= 7)
                if self.cuda:
                    weight,edge =  weight.cuda(), edge.cuda()
                maskList.append(edge)
                weightList.append(weight)
            maskEdge = torch.unsqueeze(torch.cat(maskList,axis=0),dim=0)
            weightEdge = torch.unsqueeze(torch.cat(weightList,axis=0),dim=0)
            maskEdgeList.append(maskEdge)
            weightEdgeList.append(weightEdge)

        Edge = torch.cat(maskEdgeList,axis=0)
        Weight = torch.cat(weightEdgeList,axis=0)
        logging.info(f"Weight size:{Weight.size()}")
        logging.info(f"Edge size:{Edge.size()}")
        return Edge, Weight

    def getEdgeWeight(self,IndexTensor,blurSize = 7):
        
        size = IndexTensor.size()
        logging.info(f"size of indexTensor:{size}")
        
        if len(size)==2:
            if self.cuda:
                imageMask = np.array(IndexTensor.cpu())
            else :
                imageMask = np.array(IndexTensor)
        else:
            if self.cuda:
                imageMask = np.squeeze(np.array(IndexTensor.cpu()),axis=0)
            else:
                imageMask =  np.squeeze(np.array(IndexTensor,axis=0))

        if np.all(imageMask==0):
            weight = torch.zeros(size)
            edge = torch.zeros(size)
            return edge, weight

        k = morphology.square(width = 3)      #正方形
        imageOut = morphology.erosion(imageMask, k)
        outlier = imageMask - imageOut
        edge = torch.from_numpy(outlier[None,:,:])
        blur = cv2.GaussianBlur(outlier*255,(blurSize,blurSize),0)/255.
        weight = torch.from_numpy(blur[None,:,:])
        return edge, weight

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




