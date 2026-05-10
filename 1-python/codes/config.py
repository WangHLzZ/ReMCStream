# -×- coding: utf-8 -*-
# configuration on ReMCStream

# vae+triplet+Confidence

NumInit = 1000

dict_dataset = {
    'spam': {'file':'../datasetsoforiginal/spam.csv',
             'NumClasses': 2,'h_dim': 16,'featuredim': 499,'num_k': 1,'n_epoch': 100},
    # 'yourfilename': {'file':'your_file_path.csv',
    #          'NumClasses': ,'h_dim': ,'featuredim': ,'num_k': ,'n_epoch': },
}

