function output = F1(C)
precision = diag(C) ./ sum(C,2);
recall = diag(C) ./ sum(C,1)';
output =2*(precision.*recall)./(precision+recall);
end