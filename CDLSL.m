function [Z,P] = CDLSL(Xs,Xt,Ys,Yt,options)
% Xs ¡ÊR^d*ns  The source domain samples
% Xd ¡ÊR^d*nt  The target domain samples
% Ys ¡ÊR^d*ns  The source domain labels
% Yt ¡ÊR^d*ns  The target domain labels
% options has 8 parameters:
% NeighborMode; WeightMode; k; max_iter; dim; alpha; beta; lambda
% this function only has 4 items:
% ||P*X-Z||^2F +lambda*||P||^2F +beta/2*trace(Z*Ls*Z')+alpha*||Z-(Y+B¡ÑM)||^2F

% ------------ Initialization  ---------- %
rand('seed',6666);
X=[Xs;Xt];
X=X';
[d,n] = size(X);
dim=options.dim;
M = ones(dim, n);
Z = rand(dim,n);
% ------------ Initialize P ----------------- %
options.ReducedDim = dim;
[P1,~] = PCA1(X', options);
P = P1';
% ---------- Initialize other parameters ----------------- %
linshi_St = X*X'+options.lambda*eye(d);
St2 = mpower(linshi_St,-0.5);
St3 = St2*X;

linshi_W = full(constructW(X',options));
W_graph = (linshi_W+linshi_W')*0.5;
Sum_S = sum(W_graph);
LS = diag(Sum_S)-W_graph;

inv_GS = inv(options.beta*LS+(options.alpha+1)*eye(size(LS)));

% ------------ Initialize Ytrain_pseudo ------------- %
model=svmtrain(Ys,Xs,'-s 0 -t 0 -c 1 -g 1 ');
[Ytrain_pseudo, ~, ~] = svmpredict(Yt,Xt,model);
Y=[Ys;Ytrain_pseudo];
F = Pre_label(Y);
F = F';
B = 2 * F - ones(dim, n);

%% iteration
for iter = 1:options.max_iter
    
    % ----------------update Z ----------------- %
    % Z = [alpha*(Y+B.* M)+P*X][(1+alpha)*I+beta*LS]^-1
    linshi_F = F + B.* M;
    Z = (options.alpha*linshi_F + P*X)*inv_GS;

    % -----------------update P ------------------- %
    linshi_M = St3*Z';
    linshi_M(isnan(linshi_M)) = 0;
    linshi_M(isinf(linshi_M)) = 0;
    [linshi_U,~,linshi_V] = svd(linshi_M','econ');
    linshi_U(isnan(linshi_U)) = 0;
    linshi_U(isinf(linshi_U)) = 0;
    linshi_V(isnan(linshi_V)) = 0;
    linshi_V(isinf(linshi_V)) = 0;        
    P = linshi_U*linshi_V'*St2;   %P=V*U'*St^(-1/2)£¬¶øSt2=St^(-1/2)

    % -----------------update M ------------------- %
    M = max(B.* (Z-F),0);

    % ------------- update Ytrain_pseudo ------------- %
    Zt = P*Xt';
    Zt = Zt*diag(sparse(1./sqrt(sum(Zt.^2))));
    [~,Ytrain_pseudo] = max(Zt',[],2);
    Y = [Ys;Ytrain_pseudo];
    F = Pre_label(Y);
    F = F';

    % -----------------update B ------------------- %
    
    B = 2 * F - ones(dim, n);

    % -------------- obj --------------- %
    Item1=norm(P*X-Z,'fro')^2;
    Item2=options.alpha*norm(Z-(F+B.*M),'fro')^2;    %options.alpha*norm(Z-(Y+B¡ÑM),'fro')^2
    Item3=options.lambda*norm(P,'fro')^2; 
    Item4=options.beta*trace(Z*LS*Z');
          
    obj(iter) = (Item1+Item2+Item3+Item4); 
    if iter >5 && abs(obj(iter)-obj(iter-1))<1e-3
        break;
    end
end
end