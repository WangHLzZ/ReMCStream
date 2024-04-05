    function [ Model_temp ] = remc_update_micro(Model_temp,ex,idx,currentTime,D,r)
        % 假设 model_clu, featuresUsed, closest_cluster, cpulabel, label, p_label 等变量已定义
        flag  =  D<=r;
        closest_cluster_label = Model_temp{idx,4};
        closest_cluster_labeltype = Model_temp{idx,10};

        % 微簇动态维护
        if flag
            if ex.label_flg ~= 0
                if closest_cluster_labeltype ~= -1
                    % 有标签的数据与有伪标签的Mc
                    if closest_cluster_labeltype == 0
                        if (ex.label ~= closest_cluster_label)
                            % 标签不一致处理：删旧Mc，新建一个Mc
                            Model_temp(idx,:) = [];
                            Model_temp = remc_createMC( Model_temp,ex, r,currentTime);
                        else
                            % 标签一致处理： insert、更新伪mc的L
                            Model_temp = remc_insert_ex_micro(Model_temp,ex,idx,currentTime);
%                             Model_temp{idx,10} = 1;
                        end
                    else
                    % 有标签的数据与真实标签的Mc
                        if (ex.label ~= closest_cluster_label)
                            % 标签不一致处理：删旧Mc，新建一个Mc
                            if D < (r/3)
                                Model_temp(idx,:) = [];
                            end
                            Model_temp = remc_createMC( Model_temp,ex, r,currentTime);
                        else
                            % 标签一致处理： insert
                            Model_temp = remc_insert_ex_micro(Model_temp,ex,idx,currentTime);
                        end
                    end
                else
                    % 有标签的数据与无标签的Mc：insert、赋标签、赋标签类型、赋confidence、改列表
                    Model_temp = remc_insert_ex_micro(Model_temp,ex,idx,currentTime);
%                     Model_temp{idx,4} = ex.label;
%                     Model_temp{idx,5} = 1;
%                     Model_temp{idx,10} = 1;
                    % closest_cluster.confidence = 1;                    
                end
            elseif ex.reliable_plable ~= -1
                % 逻辑与上述类似，适当调整以匹配伪标签逻辑
                if closest_cluster_labeltype ~= -1
                    % 伪标签的数据与有伪标签的Mc
                    if closest_cluster_labeltype == 0
                        if ex.reliable_plable ~= closest_cluster_label
                            % 标签不一致处理：删旧Mc，新建一个Mc
                            Model_temp(idx,:) = [];
                            Model_temp = remc_createMC( Model_temp,ex, r,currentTime);
                        else
                            % 标签一致处理：insert
                            Model_temp = remc_insert_ex_micro(Model_temp,ex,idx,currentTime);
                        end
                    else
                    % 伪标签的数据与真实标签的Mc
                        if ex.reliable_plable ~= closest_cluster_label
                            % 标签不一致处理：新建一个Mc
                            Model_temp = remc_createMC( Model_temp,ex, r,currentTime);
                        else
                            % 标签一致处理：insert
                            Model_temp = remc_insert_ex_micro(Model_temp,ex,idx,currentTime);
                        end
                    end
                else
                    % 伪标签的数据与无标签的Mc：insert
                    Model_temp = remc_insert_ex_micro(Model_temp,ex,idx,currentTime);
                end
            else
                Model_temp = remc_insert_ex_micro(Model_temp,ex,idx,currentTime);
            end
        else %数据不在最近的微簇中的情况
            Model_temp = remc_createMC( Model_temp,ex, r,currentTime);
        end
    end

