clear;
input_p=xlsread('J:\��ʦ�����ļ�\����ѧϰ��ʳƷ���\matlab����\wangpeng_ceshi�����ֻ������-����.xlsx')
%input_p=xlsread('F:\����ѧϰ��ʳƷ���\matlab����\wangpeng_zuizhong.xlsx')
%p=p.';

t=[1;1;1;1;1;1;
   1;1;1;1;1;1;
   1;1;1;1;1;1;
   1;1;1;1;1;1;
   0;0;0;0;0;0;
   0;0;0;0;0;0;
   0;0;0;0;0;0]';%24��������ݣ�16�����������
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
net_original=newff(p,t,[1,25],{'tansig', 'purelin'}, 'trainlm');%traingd,��һ��Ĵ��ݺ�����tan-sigmoid�������Ĵ��ݺ�����linear�����������ĵ�һ��Ԫ�صķ�Χ��-1��2[-1 2]�����������ĵڶ���Ԫ�صķ�Χ��0��5[0 5]��ѵ��������traingd��
%net=newff(p,t,[50,1],{TF1 TF2},'trainlm');
net_original.trainParam.show=500;%ÿ500����ʾһ�����仯���
net_original.trainParam.epochs=1000;%���ѵ������
net_original.trainParam.goal=0.00000001; %���Ҫ��
net_original.divideFcn = ''; 

net_original=train(net_original,p,t);
date=etime(clock,t1)
y=sim(net_original,p) %����p������Ƿ����Ҫ��
%test = [1 3 5]';
%test_norm = mapminmax('apply', test, inputps);Ҫ�Բ��Լ����ݽ��й�һ��
V = net_original.iw{1,1};%����㵽�м��Ȩֵ
theta1 = net_original.b{1};%�м�����Ԫ��ֵ
V=abs(V)%���ÿ�����ľ���ֵ
[sortedV,position]=sort(V,'descend') %��������
V=V.';%ת��
sumEveryColumnV=sum(V);%����ÿһ�еĺ�
VDivideSumEveryColumn=bsxfun(@rdivide,V,sumEveryColumnV);%ÿ��������ÿһ�еĺ�
W = net_original.lw{2,1};%�м�㵽�����Ȩֵ
theta2 = net_original.b{2};%��������Ԫ��ֵ
W=abs(W);
W=W.';%ת��
sumEveryColumnW=sum(W);
WDivideSumEveryColumn=bsxfun(@rdivide,W,sumEveryColumnW);
totalImpact=VDivideSumEveryColumn*WDivideSumEveryColumn;
p_test
y=sim(net_original,p_test)