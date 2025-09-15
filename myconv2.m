% function v = myconv2(a,b)
% n = size(a,1);
% v = zeros(n,n);
% 
% for i=1:n
%     for j=1:n
%         tmp = 0;
%         for k = 1:n
%             for l = 1:n
%                 ind1 = mod(i-k,n)+1;
%                 ind2 = mod(j-l,n)+1;
%                 tmp = tmp + a(ind1,ind2)*b(k,l);
%             end
%         end
%     v(i,j) = tmp;
%     end
% end



function v = myconv2(a,b)
n = size(a,1);
v = zeros(n,n);

for i=1:n
    for j=1:n
        tmp = 0;
        for k = 1:n
            for l = 1:n
                ind1 = mod(k-i,n)+1;
                ind2 = mod(l-j,n)+1;
                tmp = tmp + a(ind1,ind2)*b(k,l);
            end
        end
    v(i,j) = tmp;
    end
end