% clc; clf; clear; close all
% n = 40;
% 
% a = rand(n,n);
% b = rand(n,n);
% 
% tic;
% t1=myconv2(a,b);
% toc
% 
% tic;
% t2=ifft2(fft2(a).*fft2(b));
% toc
% 
% disp(norm(t1-t2))

clc; clf; clear; close all
n = 40;
m = 10;

a = rand(n,m);
b = rand(n,m);

tic;
t1=myconv2(a,b);
toc

tic;
t2=n*m*ifft2(ifft2(a).*fft2(b));
toc

disp(norm(t1-t2))