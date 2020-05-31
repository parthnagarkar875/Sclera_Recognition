function rt = mim(Y,sa)

[M,N] = size(Y);
x = [-6: 6];
tmp1 = exp(-(x.*x)/(2*sa*sa)); 
tmp1 = max(tmp1)-tmp1; 
ht1 = repmat(tmp1,[9 1]); 
sht1 = sum(ht1(:));
mean = sht1/(13*9);
ht1 = ht1 - mean;
ht1 = ht1/sht1;

h{1} = zeros(15,16);
for i = 1:9
    for j = 1:13
        h{1}(i+3,j+1) = ht1(i,j);
    end
end

for k=1:11
    ag = 15*k;
    h{k+1} = imrotate(h{1},ag,'bicubic','crop');
%    h{k+1} = wkeep(h{k+1},size(h{1}));
end

for k=1:12
    R{k} = conv2(Y, h{k}, 'same');
end

rt = zeros(M,N);
for i=1:M
    for j=1:N
        ER = [R{1}(i,j), R{2}(i,j), R{3}(i,j), R{4}(i,j), R{5}(i,j), R{6}(i,j),... 
                R{7}(i,j), R{8}(i,j), R{9}(i,j), R{10}(i,j), R{11}(i,j), R{12}(i,j)];
        rt(i,j) = max(ER);
    end
end

rmin = abs(min(rt(:)));
for m = 1:M
    for n = 1:N
        rt(m,n) = rt(m,n) + rmin;
    end
end
rmax = max(max(rt));
for m = 1:M
    for n = 1:N
        rt(m,n) = round(rt(m,n)*255/rmax);
    end
end