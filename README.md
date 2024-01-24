# Co-Attention-based_PPI-prediction-model
CAPPI, a deep learning model that employs co-attention mechanisms with pre-trained language models. A hallmark of CAPPI is its capability to accurately predict both PPI and PPI-sites.
## Environment
   
* python 3.9
* torch 1.10.0
* scikit-learn 1.0.1 


## Data

   The sequence embedding method is available at: [SeqVec](https://github.com/Rostlab/SeqVec)
   

## Usage
   
## PPI prediction:
       
```
   python main_ppi.py --dataid_path list.npy  --emb_path dict.npy 
```
   

## PPI-SITE prediction:
          
```
   python main_ppisite.py --dataid_path list.npy  --emb_path dict.npy  
```

## Additional settings
   
   If you wish to manually divide the validation and test sets, you can enter the addresses yourself:<br>
   --test_path<br>
   --valid_path <br>
   
   
   If you need to change the default settings, please follow the instructions below:<br>
   --epoch <br>
   --d_model <br>
   --d_k <br>
   --d_v <br>
   --d_ff <br>
   --c_layers <br>
   --n_heads <br>
   --dropout <br>
   --lr <br>

   If you want to directly test using a trained model, you can change the type to "Test", enter your test-dataset and input the pre-trained model in the main function of main_ppi.py or main_ppisite.py. <br>
   --test_path <br>
   --typet Test <br>
   
   
