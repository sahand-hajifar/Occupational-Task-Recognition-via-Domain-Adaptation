function ROD = rank_of_domain(ftAll,maSrc,target,maLabeled)
[~,~,Ps,Pt] = ftTrans_gfk2(ftAll,maSrc,target,maLabeled);
[U1,Gamma,V] = svd(Ps'*Pt);

theta = (acos(diag(Gamma)))';
s = Ps*U1;
t = Pt*V;
Ns = sum(maSrc);
Nt = length(maSrc)-Ns;
Xs = zscore(ftAll(maSrc,:));
Xt = zscore(ftAll(~maSrc,:));
optimal_d = size(Ps,2);
sigma2s = zeros(optimal_d,1);
sigma2t = zeros(optimal_d,1);

for i = 1:optimal_d
sigma2s(i) = (1/Ns)*s(:,i)'*(Xs'*Xs)*s(:,i);
sigma2t(i) = (1/Nt)*t(:,i)'*(Xt'*Xt)*t(:,i);
end

ROD = (1/optimal_d)*theta*(0.5*sigma2s./sigma2t + 0.5*sigma2t./sigma2s -1);

