function Y = Pre_label(y)
%% y:��ǩ������num_l:��ʾ�б�ǩ��������Ŀ��Y:���ɵı�ǩ����
%�����ǰѱ�ǩ����yת���ɱ�ǩ����Y
%����y�е��������ͱ�ǩΪ3����Y�ж�Ӧ��������3��Ϊ1��������Ϊ0

nClass=length(unique(y));
num=length(y);        % num1��ʾ�����ĸ���
Y_original=zeros(nClass,length(y));    % ԭʼ�ı�ǩ����ȫ��Ϊ��  

for i=1:num
    for j=1:nClass
        if j==y(i)
            Y_original(j,i)=1;   % Ϊ�б�ǩ����������ǩΪ1
        end  
    end
end
Y=Y_original';