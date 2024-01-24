## CAPPI: Deep Learning Model Based on Co-Attention Mechanism for Protein-Protein Interaction Site Prediction.

CAPPI, a deep learning model that employs co-attention mechanisms with pre-trained language models.
A hallmark of CAPPI is its capability to accurately predict both PPI and PPI-sites.

## Environment
   
   python 3.9
   torch 1.10.0
   scikit-learn 1.0.1 


## Data

   The sequence embedding method is available at: [SeqVec](https://github.com/Rostlab/SeqVec)

   

## Usage
   
## PPI prediction:
       
   '''
   python main_ppi.py --dataid_path list.npy  --emb_path dict.npy 
   
   '''
   

## PPI-SITE prediction:
          
   '''
   python main_ppisite.py --dataid_path list.npy  --emb_path dict.npy 
   
   '''

## Additional settings
   
   If you wish to manually divide the validation and test sets, you can enter the addresses yourself:
   --test_path   
   --valid_path  
   
   
   If you need to change the default settings, please follow the instructions below:
   --epoch
   --d_model
   --d_k
   --d_v
   --d_ff
   --c_layers
   --n_heads
   --dropout
   --lr 

   If you want to directly test using a trained model, you can change the type to "Test".
   --typet Test