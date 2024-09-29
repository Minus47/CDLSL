clear;
clc;
addpath('./data');
addpath('./libsvm-new');
addpath('./tool');
warning off;
for testnum=1
    scale=0.7;
    switch testnum
        case 1
            name='be-CVE';
            load('berlin_feature-test219-end');
            load('berlin_label-test219-end');
            Xs=double(feature); clear feature
            Ys=double(label);   clear label
            load('CVE_feature-test351-end');
            load('CVE_label-test351-end');
        otherwise
            break;
    end
    
    nt=size(feature,1);
    Xtrain=double(feature(1:round(nt*scale),:));
    Ytrain=double(label(1:round(nt*scale),:));
    Xtest=double(feature(round(nt*scale)+1:end,:)); clear feature
    Yreal=double(label(round(nt*scale)+1:end,:));   clear label
    
    %% normalization and PCA
    Xs=normalization(Xs',1);
    Xs=Xs';
    Xtrain=normalization(Xtrain',1);
    Xtrain=Xtrain';
    Xtest=normalization(Xtest',1);
    Xtest=Xtest';
    X=[Xs;Xtrain;Xtest];
    
    [COEFF,SCORE, latent] = pca(X);
    SelectNum = cumsum(latent)./sum(latent);
    index = find(SelectNum >= 0.98);
    pca_dim = index(1);
    X=SCORE(:,1:pca_dim);
    
    Xs = X(1:size(Xs,1),:);
    Xtrain = X(size(Xs,1)+1:size(Xs,1)+size(Xtrain,1),:);
    Xtest = X(size(Xs,1)+size(Xtrain,1)+1:end,:);
    
    Y=[Ys;Ytrain];
    numClust=length(unique(Y));
    
    options=[];
    options.NeighborMode='KNN';
    options.WeightMode='Binary';
    options.k=5;
    options.max_iter = 50;
    options.dim=numClust;
    acc=0;
    acc_max=0;

    
    %% Experiments
    p=[0.001 0.01 0.1 1 10 100 1000];
    for alpha=p
        for lambda=p
            for beta=p               
                options.alpha=alpha;
                options.lambda=lambda;
                options.beta=beta;
                
                [Z,P] = CDLSL(Xs,Xtrain,Ys,Ytrain,options);
                
                Zs=P*Xs';
                Zs = Zs*diag(sparse(1./sqrt(sum(Zs.^2))));
                Zt=P*Xtest';
                Zt = Zt*diag(sparse(1./sqrt(sum(Zt.^2))));
                [~,cls] = max(Zt',[],2);
                acc = mean(Yreal == cls)*100;

                if acc>acc_max
                    acc_max=acc;
                end

                disp([name,'      acc: ',num2str(acc),'      acc_max: ',num2str(acc_max)]);

            end
        end
    end
    
end
