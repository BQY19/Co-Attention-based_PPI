import argparse
from train_predict import *
from model import *
import torch
import torch.optim as optim
from data import *
import logging

def get_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    if not logger.handlers: 
        formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
        fileHandler = logging.FileHandler(log_file, mode='a')
        fileHandler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(fileHandler)
    return logger

def main():
    auc_max = 0
    if typet == "Train":
        train, valid, test = shuffle_split(dataid_path, trainrate=0.8, validrate=0.1, seed=args.seed)
        """train=np.load(dataid_path,allow_pickle=True).tolist()
        test=np.load(testdata,allow_pickle=True).tolist()
        valid=np.load(validdata,allow_pickle=True).tolist()
        """
        getdataloader_train = DataLoader(dataset=mydataset_site(train, embpath), batch_size=batchsize, shuffle=True,
                                         num_workers=4, drop_last=False)

        getdataloader_valid = DataLoader(dataset=mydataset_site(valid, embpath), batch_size=4, shuffle=True,
                                         num_workers=4, drop_last=False)

        getdataloader_test = DataLoader(dataset=mydataset_site(test, embpath), batch_size=4, shuffle=True,
                                        num_workers=4, drop_last=False)

        for i in range(epoch):
            logging.info("%s" % i)
            Train(model, getdataloader_train, i,lr,thre)
            valid_auc = Valid(model, getdataloader_valid,thre)
            if valid_auc > auc_max:
                auc_max = valid_auc
                Test(model, getdataloader_test,thre)

    elif typet == "Test":
        test = np.load(testdata, allow_pickle=True).tolist()
        getdataloader_test = DataLoader(dataset=mydataset_site(test, embpath), batch_size=1, shuffle=True,num_workers=4, drop_last=False)
        #model.load_state_dict(torch.load(r""))
        Test_(model, getdataloader_test)


if __name__ == '__main__':
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataid_path", type=str,default=r"/home/qingyu/modelrun/e3target_site/site_id_listmasif2000.npy", help="")
    parser.add_argument("--emb_path", type=str, default=r"/home/qingyu/modelrun/e3target_site/masif_list_3000.npy",help="")
    parser.add_argument("--test_path", type=str,default=r"", help="")
    parser.add_argument("--valid_path", type=str,default=r"", help="")
    parser.add_argument("--epoch", type=int, default=10, help="")
    parser.add_argument("--d_model", type=int, default=128, help="")
    parser.add_argument("--d_k", type=int, default=32, help="")
    parser.add_argument("--d_v", type=int, default=32, help="")
    parser.add_argument("--d_ff", type=int, default=128, help="")
    parser.add_argument("--c_layers", type=int, default=1, help="")
    parser.add_argument("--n_heads", type=int, default=8, help="")
    parser.add_argument("--dropout", type=int, default=0.2, help="")
    parser.add_argument("--lr", type=float, default=1e-3, help="")
    parser.add_argument("--seed", type=int, default=426, help="")
    parser.add_argument("--typet", type=str, default="Train", help="Train or Test")



    args = parser.parse_args()
    
    dataid_path = args.dataid_path
    embpath = args.emb_path
    typet = args.typet
    epoch = args.epoch
    seed = args.seed
    d_model=args.d_moel
    d_v=args.d_v
    d_k=args.d_k
    d_ff=args.d_f
    c_layers=args.c_layers
    n_heads=args.n_heads
    lr=args.lr
    dropout=args.dropout
    lossweight = 15
    batchsize = 32
    thre=0.3


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PPI_site(d_model, d_v, d_v, d_model, c_layers, n_heads, dropout).to(device)  # d_model=feature dim
    main()
