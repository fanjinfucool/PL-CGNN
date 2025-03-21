data = zscore(data);%归一化


[label_num1,ins_num1] = size(partial_target);
for i = 1:ins_num1
    if partial_target(6,i)==1
       partial_target(7,i)=1;
    end
end

rand=randperm(size(data,1));%将数据打乱
data = data(rand,:);
partial_target = partial_target(:,rand);
target = target(:,rand); 


ker  = 'rbf'; %Type of kernel function

par  =1*mean(pdist(data)); %Parameters of kernel function
train_data=[];
train_p_target=[];
test_data=[];
test_target=[];
nfold = length(tr_idx);
premarx={};
    

step = nfold/100;
count = 0;
steps = 100/nfold;
fprintf('0%%');
fprintf(repmat('>',1,50));
fprintf('100%%\n');
fprintf('0%%');

for i=1:nfold%10折交叉验证，每一折都要训练一次
    if rem(i,step) < 1
        fprintf(repmat('\b',1,count-1));
        fprintf(repmat('>',1,4));
        count = fprintf(1,'>%d%%',round((i)*steps));
    end
    
    train_data = data(tr_idx{i},:);
    train_p_target = partial_target(:,tr_idx{i});
    test_data = data(te_idx{i},:);
    test_target = target(:,te_idx{i});
    
   % [Beta,b] =PL_CGNN_train(train_data,train_p_target,ker,par,0.90,20,0.26,3);
    [Beta,b] =PL_CGNN_train(train_data,train_p_target,ker,par,0.90,20,0.26,1);
    
    [Outputs,Pre_Labels,Accuracy]= PL_CGNN_predict(train_data,test_data,test_target,ker,Beta,b,par);
    acc(i)=Accuracy;
end
fprintf('\n');


average_acc=mean(acc); %求均值
    


    ss=std(acc);
disp(['准确率为   ',num2str(average_acc)])
disp(['标准差为   ',num2str(ss)])
