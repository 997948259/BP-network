clear;
input_p=xlsread('J:\陕师大工作文件\机器学习与食品结合\matlab程序\wangpeng_ceshi排序后只有酯类-新新.xlsx')
%input_p=xlsread('F:\机器学习与食品结合\matlab程序\wangpeng_zuizhong.xlsx')
%p=p.';

t=[1;1;1;1;1;1;
   1;1;1;1;1;1;
   1;1;1;1;1;1;
   1;1;1;1;1;1;
   0;0;0;0;0;0;
   0;0;0;0;0;0;
   0;0;0;0;0;0]';%24个蟠桃数据，16个非蟠桃数据
%[pn,minp,maxp,tn,mint,maxt]=premnmx(p,t)
[p,inputps]=mapminmax(input_p);
p_test=cat(2,p(:,21:24),p(:,39:42));
p=cat(2,p(:,1:20),p(:,25:38));
t=cat(2,t(:,1:20),t(:,25:38))
t1=clock;
TF1 = 'tansig';
TF2 = 'purelin'; 
%TF2 = 'logsig';
%net_original=newff(p,t,[25,1],{TF1 TF2},'trainlm');
net_original=newff(p,t,[1,25],{'tansig', 'purelin'}, 'trainlm');%traingd,第一层的传递函数是tan-sigmoid，输出层的传递函数是linear。输入向量的第一个元素的范围是-1到2[-1 2]，输入向量的第二个元素的范围是0到5[0 5]，训练函数是traingd。
%net=newff(p,t,[50,1],{TF1 TF2},'trainlm');
net_original.trainParam.show=500;%每500次显示一下误差变化情况
net_original.trainParam.epochs=1000;%最大训练次数
net_original.trainParam.goal=0.00000001; %误差要求
net_original.divideFcn = ''; 

net_original=train(net_original,p,t);
date=etime(clock,t1)
y=sim(net_original,p) %测试p的输出是否符合要求
%test = [1 3 5]';
%test_norm = mapminmax('apply', test, inputps);要对测试集数据进行归一化
V = net_original.iw{1,1};%输入层到中间层权值
theta1 = net_original.b{1};%中间层各神经元阈值
V=abs(V)%获得每个数的绝对值
[sortedV,position]=sort(V,'descend') %醇类物质
V=V.';%转置
sumEveryColumnV=sum(V);%计算每一列的和
VDivideSumEveryColumn=bsxfun(@rdivide,V,sumEveryColumnV);%每个数除以每一列的和
W = net_original.lw{2,1};%中间层到输出层权值
theta2 = net_original.b{2};%输出层各神经元阈值
W=abs(W);
W=W.';%转置
sumEveryColumnW=sum(W);
WDivideSumEveryColumn=bsxfun(@rdivide,W,sumEveryColumnW);
totalImpact=VDivideSumEveryColumn*WDivideSumEveryColumn;
p_test
y=sim(net_original,p_test)