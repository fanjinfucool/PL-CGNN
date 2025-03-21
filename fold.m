
nfold = 10;

[n_sample,n_fea]= size(data);%样本的个数,特征维度

n_test = round(n_sample/nfold);%测试样本的个数，round函数为取整，四舍五入

I = 1:n_sample;
te_idx={};
tr_idx={};
for i=1:nfold%多折交叉验证，每一折都要训练一次
    fprintf('data2 processing,Cross validation: %d\n', i);
    start_ind = (i-1)*n_test + 1;
    if start_ind+n_test-1 > n_sample
        test_ind = start_ind:n_sample;
    else
        test_ind = start_ind:start_ind+n_test-1;
    end
    train_ind = setdiff(I,test_ind);
    
    tr_idx{i}=train_ind;
    te_idx{i}=test_ind;
    
end

tr_idx=tr_idx';
te_idx=te_idx';

