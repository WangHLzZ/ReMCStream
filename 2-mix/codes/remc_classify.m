function [ final_label, idxf,Dist] = remc_classify( ex, Model, num_of_k)
    
    clu_cent=cell2mat(Model(:,6));
    [idx_1, D_1]=knnsearch(clu_cent,ex.data,'NSMethod','exhaustive','k',1);
    if cell2mat(Model(idx_1,5))==1 && D_1<=Model{idx_1,7}
        p_label = cell2mat(Model(idx_1,4));
        idxn = idx_1;
        Dist = D_1;
   
    else
        label_clu_cen=find(cell2mat(Model(:,5))~=0); % 可能需要修改，又或许不用
        clu_cent=clu_cent(label_clu_cen,:);
        % knn classifier
        [idx, D]=knnsearch(clu_cent,ex.data,'NSMethod','exhaustive','k',num_of_k);
        Dist = D;
        idxn=label_clu_cen(idx);
        nn_label=cell2mat(Model(idxn,4));
        [uninn_cls_lb, ia2, cid2] = unique(nn_label,'stable');
        counts2 = sum( bsxfun(@eq, cid2, unique(cid2)') )';
        [Y I]=max(counts2);
        p_label=uninn_cls_lb(I);
  
    end
    idxf=idxn;
    final_label=p_label;

end

