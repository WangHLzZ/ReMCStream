function [dataCells] = remc_addDataToMatrix(dataCells, data, labelIdx, maxRows)
% data: 要添加的数据，假设是一个行向量
    % labelIdx: 数据对应的类别索引
    % maxRows: 矩阵的最大行数
    % 获取指定类别当前的矩阵
    currentMatrix = dataCells{labelIdx};
    % 添加数据到矩阵的末尾
    updatedMatrix = [currentMatrix; data];
    % 检查是否超出最大行数，如果是，则删除第一行
    if size(updatedMatrix, 1) > maxRows
        updatedMatrix = updatedMatrix(2:end, :); % 删除第一行
    end
    % 更新 cell array 中的矩阵
    dataCells{labelIdx} = updatedMatrix;
end

