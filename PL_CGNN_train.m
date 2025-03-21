function [Beta,b]= PL_CGNN_train(train_data,train_p_target,ker,par,alpha,maxIter,pi,K)

trainData=train_data;
trainTarget = train_p_target;
trainData = normr(trainData); %数据归一化
[label_num,ins_num] = size(trainTarget);
y = trainTarget';
fea_num = size(trainData,2);
W1 = zeros(fea_num,label_num);

for i = 1:label_num
    k = find(y(:,i)==1);
    w = trainData(k,:);
    %w = sgolayfilt(w',1,5);
    w = sgolayfilt(w',4,13);
    w = mean(w,2); %按行平均
    W1(:,i)=w;
end

W2 = zeros(fea_num,label_num);

label_sum = sum(y,2);
label_k = find(label_sum == 1);
label_y = y(label_k,:);

for i = 1:label_num
    if sum(label_y(:,i),1)==0
        W2(:,i) = W1(:,i);
    else       
        k = find(label_y(:,i)==1);
        w = trainData(k,:);
        w = mean(w',2); 
        W2(:,i) = w;
    end
end

k=10;
kdtree = KDTreeSearcher(trainData);
[neighbor,dist] = knnsearch(kdtree,trainData,'k',k+1);
neighbor = neighbor(:,2:k+1);

%%
%自适应神经网络
%competitive_network
% alpha = 0.65;
iter=0;
% maxIter=50;

while iter<maxIter
    iter=iter+1;
 
    real_label = zeros(ins_num,label_num);
    for i = 1:ins_num
        %前向传播
        u = zeros(1,label_num);
        a = trainData(i,:)';
        k = find(y(i,:)==1);
        [D,d,real_cloumn] = distance(W1,W2,a,k,pi);
%         real_label(i,real_cloumn)=1;
        u(:,real_cloumn)=u(:,real_cloumn)+1;
        kd = 1-d;
        for j = 1:K
            b = trainData(neighbor(i,j),:)';
            k = find(y(neighbor(i,j),:)==1);
            [D,d,real_cloumn] = distance(W1,W2,b,k,pi);
            u(:,real_cloumn)=u(:,real_cloumn)+1;
        end
        u=u.*y(i,:);
        [maxVal maxInd] = max(u);
         real_label(i,maxInd)=1;
        %反向传播
%         W1(:,real_cloumn) = W1(:,real_cloumn)-alpha*(W1(:,real_cloumn)-a);
        W1(:,real_cloumn) = W1(:,real_cloumn)-kd*alpha*(W1(:,real_cloumn)-a);
%         W1(:,real_cloumn) = normr(W1(:,real_cloumn));
    end
    for i = 1:label_num
        k = find(real_label(:,i)==1);
        w = trainData(k,:);
%         w = sgolayfilt(w',1,5);
        w = mean(w',2); %按行平均
        W2(:,i)=w;
    end
end
y = real_label;
C1=10; %Penalty parameter
C2=1; %Penalty parameter

tol  = 1e-10; %Tolerance during the iteration
epsi =0.1; %Instances whose distance computed is more than epsi should be penalized

[Beta,b] = plmsvr(train_data,y,train_p_target',ker,C1,C2,epsi,par,tol);
end
function [D,d,real_cloumn] = distance(W1,W2,a,k,pi)
W11 = W1(:,k);
W22 = W2(:,k);
% D = 0.35*sum((W11-a).^2).^0.5+(1-0.3)*sum((W22-a).^2).^0.5;%欧式距离
D = pi*sum((W11-a).^2).^0.5+(1-pi)*sum((W22-a).^2).^0.5;%欧式距离
d = exp(-D);
[maxVal maxInd] = max(d);
d = maxVal;
real_cloumn = k(:,maxInd);
end

