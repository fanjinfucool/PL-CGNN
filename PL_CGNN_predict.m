function [Outputs,Pre_Labels,Accuracy]= PL_CGNN_predict(train_data,test_data,test_target,ker,Beta,b,par)
num_test=size(test_data,1); %the number of testing instance
test_target=test_target';
num_label=size(test_target,2);
Pre_Labels=zeros(num_test,num_label);
Outputs=zeros(num_test,num_label);
count=0;
Ktest = kernelmatrix(ker,test_data',train_data',par);
Ypredtest =Ktest*Beta+repmat(b,num_test,1);
for j=1:num_test
    distribution = Ypredtest(j,:);
    Outputs(j,:)=distribution;
    [~,class]=max(distribution);
    Pre_Labels(j,class)=1;
    if(test_target(j,class)==1)
        count=count+1; 
    end
end
Outputs=Outputs';
Pre_Labels=Pre_Labels';
Accuracy=count/num_test;
end


