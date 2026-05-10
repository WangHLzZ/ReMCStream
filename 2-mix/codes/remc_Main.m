function remc_Main(dsname,labelratio)
    
    % 获取 数据集 的配置信息
%     datasetname = 'arbf';
    datasetname = dsname;
    configDataset = remc_configure(datasetname);
    % spam % elec % gsd % shuttle % sldd % occupancy % weather % fct %
    % waveg % waves % wavesandr % arbf % atree % asea % 
%     disp(configDataset);
    %%%%%%%%%%%%%%%%%%%% init data %%%%%%%%%%%%%%%%%%%%%%%%%%%
    npyfeatureFilePath = ['../datasets/', configDataset.featurePath];
    npyData = py.numpy.load(npyfeatureFilePath);
    npylabelFilePath = ['../datasets/', configDataset.labelPath];
    npyLabel = py.numpy.load(npylabelFilePath);
    int_data=double(npyData); % n * h
    int_label = double(npyLabel); % 1*n
    %%%%%%%%%%%%%%%%%%%% stream %%%%%%%%%%%%%%%%%%%%%%%%%%%
    filename = ['../datasetsoforiginal/', configDataset.originDataPath];
    orgindata = readtable(filename, 'HeaderLines', 1);
    stream_data = table2array(orgindata(:, :));
    % 初始化python module
    encodernet = py.importlib.import_module('encoder_predict4matlab');
    py.importlib.reload(encodernet);
   
    confnet = py.importlib.import_module('confnet_predict4matlab');
    py.importlib.reload(confnet);
    out1 = confnet.confnet_predict( ...
        pyargs('lowfeature2predict',npyData,'labels',npyLabel, 'datasetname',datasetname));
    
    confnetUpdate = py.importlib.import_module('confnet_update4matlab');
    py.importlib.reload(confnetUpdate);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Important var Parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    global maxMC; %maximum number of micro-clusters
    maxMC=1000;
    global num_cluster; %number of clusters per class
    num_cluster=50;
    global lamda; % decay rate
%     lamda=0.000002;% 000002[2K 0.06]
    lamda=0.00001;
    global D_init;
    D_init=1000; %initial training data
    global wT;
%     wT=0.06;
    wT=0.000001;
    k = 1;
    rng(2024);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     for ij=1:4 % Label ratio
    for ij=labelratio
        time_all_start = tic;
        conflistArray = cell(out1{2});
        confidence_list = cellfun(@double, conflistArray);
        c_threshold = out1{1};
        BufferSize = 10*configDataset.num_class;
        EachClassBufferSize = ceil(BufferSize / configDataset.num_class);
        fprintf('Label Percentage =%d  \n',labelratio);

        if length(confidence_list) > BufferSize
            confidence_list = confidence_list(end-BufferSize+1:end);
            assert(length(confidence_list) == BufferSize);
        end
        tau = 0.4;
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        label_per=labelratio;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%initial training data D_init %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%         train_data=stream_data(1:D_init,:);
%         train_data_labels=train_data(:,end);
        train_data=[int_data,int_label'];
        train_data_labels=int_label';

        test_data=stream_data(D_init+1:end,1:end-1);
        test_data_labels=stream_data(D_init+1:end,end);
        [train_cls_lb, ia2, cid2] = unique(train_data_labels);
        train_cls_lb = sort(train_cls_lb);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [Model,labeledData]=remc_initial_model_construction( ...
            train_data,train_cls_lb,configDataset.num_class,EachClassBufferSize);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %percentage of partial labels
        test_size=size(test_data,1); %size of test stream

        no_label_data=ceil(label_per*test_size/100);
        rno=randperm(test_size,no_label_data);
        test_data_labels=[test_data_labels zeros(1,test_size)'];
        test_data_labels(rno,2)=1;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        acc=[];
        correct=0;
        j=1;
        i=1;
        numofnetupdate = 0;
        time_predict_sum = 0;
        time_update_sum = 0;
        while i<=test_size
            %disp(b_no); hidden_data,confpredidx,confidence
%             py.importlib.reload(encodernet);
            time_predict_start = tic;
            netoutput = encodernet.encoderandconf_predict(pyargs( ...
                'data2predict',test_data(i,:),'datasetname',datasetname,'numUpdate',numofnetupdate));
            time_predict_end = toc(time_predict_start);
            time_predict_sum = time_predict_sum + time_predict_end;
 %             ex.data=test_data(i,:);
%             ex.data=double(hidden_data);
%             disp(netoutput{2})
%             disp(netoutput{3})
            predict_idx = double(netoutput{2});
            conf_of_confnetwork_used = double(netoutput{3});
            ex.data=double(netoutput{1});
            ex.label=test_data_labels(i,1);
            ex.label_flg=test_data_labels(i,2);
            ex.reliable_plable = -1; %label_flg为1时，rp必为-1；label_flg不为1时，rp不一定为-1
            CurrentTime=i;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [ p_label, idx, dist] = remc_classify( ex, Model, k );

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %对于无标签数据计算其可靠性score
            score = 0;
            if (p_label == predict_idx) && conf_of_confnetwork_used >= c_threshold
                for idx_k=1:k
                    k_idx = idx(idx_k);
                    %%part1
                    eT = (CurrentTime-cell2mat(Model(k_idx,8)));
                    part1 = exp(-0.01 * eT);
                    %%part2
                    r_k = Model{k_idx,7};
                    if dist(idx_k) <= r_k
                        part2 = 1;
                    else
                        part2 = r_k/dist(idx_k);
                    end
                    %%part3
                    part3 = 1 / (1 + exp(-1 * cell2mat(Model(k_idx,9))));
                    %%total
                    if cell2mat(Model(k_idx,4)) == p_label
                        factor = 1;
                    else
                        factor = -1;
                    end
                    score = score + factor * part1 * part2 * part3;
                end
                score = score / k;
            end

            
            if score>=tau && ex.label_flg~=1
                ex.reliable_plable = p_label;
            end

            if p_label==ex.label
                correct=correct+1;
            end

            FlagUpdateConfnet = false;
            if ex.label_flg==1
                % 向buffer中添加有标签的数据
                labeledData = remc_addDataToMatrix( ...
                    labeledData, ex.data, ex.label+1, EachClassBufferSize);
                %判断是否需要更新网络
                if (predict_idx ~= ex.label) && (conf_of_confnetwork_used >= c_threshold)
                    FlagUpdateConfnet = true;
                end
                %对tau进行动态更新
                if p_label == predict_idx
                    if (ex.label == p_label) && (score < tau)
                        tau = tau * (1 - 0.01);
                    elseif (ex.label ~= p_label) && (score >= tau)
                        tau = tau * (1 + 0.01);
                    end
                end
                %微簇代表性维护
                nrb_labels=cell2mat(Model(idx,4));
                con_label=idx(nrb_labels==ex.label);
                incon_label=idx(nrb_labels~=ex.label);
                Model(con_label,9)=num2cell(cell2mat(Model(con_label,9))+1); %consistant with true label
                Model(con_label,8)=num2cell(CurrentTime);
                Model(incon_label,9)=num2cell(cell2mat(Model(incon_label,9))-(1/k)); %inconsistant with true label

            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            [ Model] = remc_update_Model( Model, CurrentTime);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            clu_cent=cell2mat(Model(:,6));
            [idx, D]=knnsearch(clu_cent,ex.data,'NSMethod','exhaustive');
            r=Model{idx,7};
            [ Model ] = remc_update_micro(Model,ex,idx, CurrentTime,D,r);

%             if D<=r && ex.label_flg==1 && ex.label==cell2mat(Model(idx,4))||D<=r && ex.label_flg~=1
%                 [ Model ] = update_micro(Model,ex,idx, CurrentTime);
%             else
%                 [ Model] = createMC( Model,ex, r,CurrentTime);
%             end

            acc(j,1)=correct*100/i;
            if mod(i,1000)==0
                fprintf('\n example no =%d',i);
                fprintf('\t accuracy=%f  \n',acc(j,1));
            end
            i=i+1;
            j=j+1;

            if FlagUpdateConfnet

                labelVector = [];
                trainData = [];

                for idx_ld = 1:length(labeledData)
                    numRows = size(labeledData{idx_ld}, 1);
                    trainData = [trainData;labeledData{idx_ld}];
                    currentLabels = repmat(idx_ld-1, 1, numRows);
                    labelVector = [labelVector, currentLabels];
                end

                labelVector = labelVector';
                time_update_start = tic;
                confnetUpdate.confnet_update(pyargs('labelData4train',trainData, ...
                    'labels',labelVector,'datasetname',datasetname,'numUpdate',numofnetupdate))
                time_update_end = toc(time_update_start);
                time_update_sum = time_update_sum + time_update_end;
                numofnetupdate = numofnetupdate+1;
            end

            if length(confidence_list) >= BufferSize
                confidence_list(1) = []; 
                confidence_list = [confidence_list, conf_of_confnetwork_used];
                assert(length(confidence_list) == BufferSize, 'Length of confidence_list does not equal BufferSize.');
            else
                confidence_list = [confidence_list, conf_of_confnetwork_used];
            end
            
            c_mean = mean(confidence_list);
            c_std = std(confidence_list); 
            c_threshold = c_mean + c_std;
        end

        elapsed_time = toc(time_all_start);

        fprintf('\t Label Percentage =%d  \n',labelratio);
        fprintf('\t Final Accuracy=%f  \n \n',acc(end));
        dset_f_acc(ij)=acc(end);
    end
end
