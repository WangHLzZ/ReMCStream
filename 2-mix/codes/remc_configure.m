function config = remc_configure(datasetName)
    config = struct();

    switch datasetName
        case 'spam'
            config.featurePath = 'spam_features_16hidden_100epoch.npy';
            config.labelPath = 'spam_labels_16hiden_100epoch.npy';
            config.originDataPath = 'spam.csv';
            config.encoderSavePath = 'spam_final_model_16hidden_100epoches.pth';
            config.confnetSavePath = 'spam_final_conf_model_16hidden_100epoche.pth';
            config.h_dim = 16;
            config.num_class = 2;
            config.feature_dim = 499;
            
        otherwise
            error('Unknown dataset name: %s', datasetName);
    end
end
