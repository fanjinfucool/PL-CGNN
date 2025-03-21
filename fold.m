
nfold = 10;

[n_sample,n_fea]= size(data);%�����ĸ���,����ά��

n_test = round(n_sample/nfold);%���������ĸ�����round����Ϊȡ������������

I = 1:n_sample;
te_idx={};
tr_idx={};
for i=1:nfold%���۽�����֤��ÿһ�۶�Ҫѵ��һ��
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

