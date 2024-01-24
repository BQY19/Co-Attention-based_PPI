import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import logging
from model import *
from data import *
from main_ppi import *

def loss_fun(prob,target,mask1,mask2):
    loss_fn = nn.BCELoss(reduce=True)
    mask=torch.cat([mask1,mask2],dim=1)
    posten=target*15
    posten=posten.add(1)
    loss=torch.sum(loss_fn(prob,target)*mask*posten)/torch.sum(mask)
    return loss



def Train(model, train_loader, epoch,lr,thre):
    log_file1 = 'train.log'
    logtrain = get_logger('logtrain', log_file1)
    logtrain.info("training")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    label_train = []
    loss_sum = []
    problist = []
    predlist_train = []
    num = 0
    for inputemb1, inputmask1, inputemb2, inputmask2, target, _, _ in train_loader:  

        inputemb1 = inputemb1.to(device)
        inputemb2 = inputemb2.to(device)
        inputmask1 = inputmask1.to(device)
        inputmask2 = inputmask2.to(device)
        target1 = target.to(device)
        
        optimizer.zero_grad()
        output1, _, _ = model(inputemb1, inputemb2, inputmask1, inputmask2)  # [batch_size ,1000, 1]
        output = output1.squeeze(-1)
        target = target1.squeeze(-1)
        # recordfile.write(f"{output}\n{target}\n")
        loss = loss_fun(output, target, inputmask1, inputmask2)  #
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=2)
        optimizer.step()
        # tensor.cpu().detach().numpy()
        loss_sum.append(loss.item())
        prob = output.detach().cpu()
        pred = getBinaryTensor(prob, thre)
        prob = prob.numpy().tolist()
        pred = pred.numpy().tolist()
        target = target.detach().cpu().numpy().tolist()

        for i in range(len(prob)):
            problist.extend(prob[i])
            predlist_train.extend(pred[i])
            label_train.extend(target[i])

    AUC_train = roc_auc_score(y_true=label_train, y_score=problist)
    loss_mean_train = np.mean(loss_sum)
    ACC = accuracy_score(label_train, predlist_train)
    f1 = f1_score(label_train, predlist_train)
    Precision = precision_score(label_train, predlist_train)
    Recall = recall_score(label_train, predlist_train)
    logtrain.info(
        f"epoch：{epoch}\nAUC_train：{AUC_train}\nloss_mean_train：{loss_mean_train}\nACC：{ACC}\nf1：{f1}\nPrecision：{Precision}\nRecal：{Recall}\nloss_mean：{loss_mean_train}\n")

def Train_ppi(model, train_loader, epoch,lr,thre):
    log_file1 = 'train.log'
    logtrain = get_logger('logtrain', log_file1)
    logtrain.info("training")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.BCELoss(reduce=True)
    model.train()
    label_train = []
    loss_sum = []
    problist = []
    predlist_train = []
    num = 0
    for inputemb1, inputmask1, inputemb2, inputmask2, target, _, _ in train_loader:  

        inputemb1 = inputemb1.to(device)
        inputemb2 = inputemb2.to(device)
        inputmask1 = inputmask1.to(device)
        inputmask2 = inputmask2.to(device)
        target1 = target.to(device)
        
        optimizer.zero_grad()
        output1, _, _ = model(inputemb1, inputemb2, inputmask1, inputmask2)  # [batch_size ,1000, 1]
        output = output1.squeeze(-1)
        target = target1.squeeze(-1)
        # recordfile.write(f"{output}\n{target}\n")
        loss = loss_fn(output, target)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=2)
        optimizer.step()
        # tensor.cpu().detach().numpy()
        loss_sum.append(loss.item())
        prob = output.detach().cpu()
        pred = getBinaryTensor(prob, thre)
        prob = prob.numpy().tolist()
        pred = pred.numpy().tolist()
        target = target.detach().cpu().numpy().tolist()
        
        problist.extend(prob)
        predlist_train.extend(pred)
        label_train.extend(target)


    AUC_train = roc_auc_score(y_true=label_train, y_score=problist)
    loss_mean_train = np.mean(loss_sum)
    ACC = accuracy_score(label_train, predlist_train)
    f1 = f1_score(label_train, predlist_train)
    Precision = precision_score(label_train, predlist_train)
    Recall = recall_score(label_train, predlist_train)
    logtrain.info(
        f"epoch：{epoch}\nAUC_train：{AUC_train}\nloss_mean_train：{loss_mean_train}\nACC：{ACC}\nf1：{f1}\nPrecision：{Precision}\nRecal：{Recall}\nloss_mean：{loss_mean_train}\n")


def Valid(model, valid_loader,thre):  
    log_file3 = 'record.log'
    logrecord = get_logger('logrecord', log_file3)
    logrecord.info("validing")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    label_valid = []
    problist = []
    predlist_valid = []
    with torch.no_grad():
        for inputemb1, inputmask1, inputemb2, inputmask2, target, _, _ in valid_loader:  # input: ("123456789","123456789") → (pro1,pro2)
            inputemb1 = inputemb1.to(device)
            inputemb2 = inputemb2.to(device)
            inputmask1 = inputmask1.to(device)
            inputmask2 = inputmask2.to(device)
            target1 = target.to(device)
            output1, _, _ = model(inputemb1, inputemb2, inputmask1, inputmask2)

            output = output1.squeeze_(-1)
            target = target1.squeeze_(-1)

            prob = output.detach().cpu()
            pred = getBinaryTensor(prob, thre)
            prob = prob.numpy().tolist()

            pred = pred.numpy().tolist()
            target = target.detach().cpu().numpy().tolist()
            
            for i in range(len(prob)):
                predlist_valid.extend(pred[i])
                problist.extend(prob[i])
                label_valid.extend(target[i])

    AUC_valid = roc_auc_score(y_true=label_valid, y_score=problist)
    ACC = accuracy_score(label_valid, predlist_valid)
    f1 = f1_score(label_valid, predlist_valid)
    Precision = precision_score(label_valid, predlist_valid)
    Recall = recall_score(label_valid, predlist_valid)
    logrecord.info(
        f"AUC_valid：{AUC_valid}\nACC：{ACC}\nf1：{f1}\nPrecision：{Precision}\nRecall：{Recall}\n---------------------")
    return AUC_valid

def Valid_ppi(model, valid_loader,thre):  
    log_file3 = 'record.log'
    logrecord = get_logger('logrecord', log_file3)
    logrecord.info("validing")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    label_valid = []
    problist = []
    predlist_valid = []
    with torch.no_grad():
        for inputemb1, inputmask1, inputemb2, inputmask2, target, _, _ in valid_loader: 
            inputemb1 = inputemb1.to(device)
            inputemb2 = inputemb2.to(device)
            inputmask1 = inputmask1.to(device)
            inputmask2 = inputmask2.to(device)
            target1 = target.to(device)
            output1, _, _ = model(inputemb1, inputemb2, inputmask1, inputmask2)

            output = output1.squeeze_(-1)
            target = target1.squeeze_(-1)
            prob = output.detach().cpu()
            pred = getBinaryTensor(prob, thre)
            pred = pred.numpy().tolist()
            target = target.detach().cpu().numpy().tolist()
            problist.extend(prob)
            predlist_valid.extend(pred)
            label_valid.extend(target)


    AUC_valid = roc_auc_score(y_true=label_valid, y_score=problist)
    ACC = accuracy_score(label_valid, predlist_valid)
    f1 = f1_score(label_valid, predlist_valid)
    Precision = precision_score(label_valid, predlist_valid)
    Recall = recall_score(label_valid, predlist_valid)
    logrecord.info(
        f"AUC_valid：{AUC_valid}\nACC：{ACC}\nf1：{f1}\nPrecision：{Precision}\nRecall：{Recall}\n---------------------")
    return AUC_valid



def Test(model, test_loader,thre): 
    log_file2 = 'test.log'
    logtest = get_logger('logtest', log_file2)
    logtest.info("testing")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_list = []
    label = []
    problist = []
    predlist = []
    test_pred = []
    test_target = []
    num = 0
    with torch.no_grad():
        for inputemb1, inputmask1, inputemb2, inputmask2, target, id1, id2 in test_loader:  
            inputemb1 = inputemb1.to(device)
            inputemb2 = inputemb2.to(device)
            inputmask1 = inputmask1.to(device)
            inputmask2 = inputmask2.to(device)  
            target1 = target.to(device)
            model.eval()
            output1, hieattn1, hieattn2 = model(inputemb1, inputemb2, inputmask1,
                                                inputmask2) 
            output = output1.squeeze(-1)
            target = target1.squeeze(-1)
            hieattn1 = hieattn1.detach().cpu().numpy()
            hieattn2 = hieattn2.detach().cpu().numpy()

            loss = loss_fun(output, target, inputmask1, inputmask2)
            loss_list.append(loss.item())
            prob = output.detach().cpu()
            pred = getBinaryTensor(prob, thre)
            prob = prob.numpy().tolist()
            pred = pred.numpy().tolist()
            target = target.detach().cpu().numpy().tolist()
            for i in range(len(prob)):
                problist.extend(prob[i])
                predlist.extend(pred[i])
                label.extend(target[i])  #

    ACC = accuracy_score(label, predlist)
    Precision = precision_score(label, predlist)
    Recall = recall_score(label, predlist)
    mean_loss = np.mean(loss_list)
    f1 = f1_score(label, predlist)  
    AUC = roc_auc_score(y_true=label, y_score=problist)  
    test_target.extend(label)
    test_pred.extend(problist)
    #torch.save(model.state_dict(), "")
    logtest.info(
        f"AUC_test：{AUC}\nACC：{ACC}\nf1：{f1}\nPrecision：{Precision}\nRecall：{Recall}\n")


def Test_ppi(model, test_loader,thre): 
    log_file2 = 'test.log'
    logtest = get_logger('logtest', log_file2)
    logtest.info("testing")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_list = []
    label = []
    problist = []
    predlist = []
    test_pred = []
    test_target = []
    num = 0
    with torch.no_grad():
        for inputemb1, inputmask1, inputemb2, inputmask2, target, id1, id2 in test_loader:  
            inputemb1 = inputemb1.to(device)
            inputemb2 = inputemb2.to(device)
            inputmask1 = inputmask1.to(device)
            inputmask2 = inputmask2.to(device)  
            target1 = target.to(device)
            model.eval()
            output1, hieattn1, hieattn2 = model(inputemb1, inputemb2, inputmask1,
                                                inputmask2) 
            output = output1.squeeze(-1)
            target = target1.squeeze(-1)
            hieattn1 = hieattn1.detach().cpu().numpy()
            hieattn2 = hieattn2.detach().cpu().numpy()


            prob = output.detach().cpu()
            pred = getBinaryTensor(prob, thre)
            prob = prob.numpy().tolist()
            pred = pred.numpy().tolist()
            target = target.detach().cpu().numpy().tolist()

            problist.extend(prob)
            predlist.extend(pred)
            label.extend(target) 

    ACC = accuracy_score(label, predlist)
    Precision = precision_score(label, predlist)
    Recall = recall_score(label, predlist)
    f1 = f1_score(label, predlist)  
    AUC = roc_auc_score(y_true=label, y_score=problist)  
    test_target.extend(label)
    test_pred.extend(problist)
    #torch.save(model.state_dict(), "")
    logtest.info(
        f"AUC_test：{AUC}\nACC：{ACC}\nf1：{f1}\nPrecision：{Precision}\nRecall：{Recall}\n")

def Test_(model, test_loader,thre):
    log_file2 = 'test.log'
    logtest = get_logger('logtest', log_file2)
    logtest.info("testing")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_list = []
    label = []
    problist = []
    predlist = []
    num = 0
    with torch.no_grad():
        for inputemb1, inputmask1, inputemb2, inputmask2, target, id1, id2, seq1, seq2 in test_loader:  
            inputemb1 = inputemb1.to(device)
            inputemb2 = inputemb2.to(device)
            inputmask1 = inputmask1.to(device)
            inputmask2 = inputmask2.to(device)  
            target1 = target.to(device)
            model.eval()
            output1, hieattn1, hieattn2 = model(inputemb1, inputemb2, inputmask1,
                                                inputmask2)  
            output = output1.squeeze(-1)
            target = target1.squeeze(-1)
            hieattn1 = hieattn1.detach().cpu().numpy()
            hieattn2 = hieattn2.detach().cpu().numpy()
            loss = loss_fun(output, target, inputmask1, inputmask2)
            loss_list.append(loss.item())
            prob = output.detach().cpu()
            pred = getBinaryTensor(prob, thre)
            prob = prob.numpy().tolist()
            pred = pred.numpy().tolist()
            target = target.detach().cpu().numpy().tolist()
            for i in range(len(prob)):
                problist.extend(prob[i])
                predlist.extend(pred[i])
                label.extend(target[i])  #

            logrecord.info(f"{id1}\n")
            for j in range(len(seq1[0])):
                logrecord.info(f"{j}\t{seq1[0][j]}\t{target[0][j]}\t{prob[0][j]}")
            logrecord.info(f"\n{id2}\n")
            for j in range(len(seq2[0])):
                logrecord.info(f"{j}\t{seq2[0][j]}\t{target[0][766 + j]}\t{prob[0][766 + j]}")

            ACC = accuracy_score(label, predlist)
            Precision = precision_score(label, predlist)
            Recall = recall_score(label, predlist)
            mean_loss = np.mean(loss_list)
            f1 = f1_score(label, predlist)
            AUC = roc_auc_score(y_true=label, y_score=problist)

            logtest.info(f"{id1}\t{id2}\t{AUC}\t{f1}\t{Precision}\t{Recall}\t{ACC}\n")

    ACC = accuracy_score(label, predlist)
    Precision = precision_score(label, predlist)
    Recall = recall_score(label, predlist)
    mean_loss = np.mean(loss_list)
    f1 = f1_score(label, predlist)
    AUC = roc_auc_score(y_true=label, y_score=problist)
    test_target.extend(label)
    test_pred.extend(problist)
    logtest.info(
        f"AUC_test：{AUC}\nACC：{ACC}\nf1：{f1}\nPrecision：{Precision}\nRecall：{Recall}\nloss_mean：{mean_loss}\n---------------------")

