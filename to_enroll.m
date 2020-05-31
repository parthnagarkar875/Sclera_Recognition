clc;
clear all;
close all;
warning off all;

%% Inputs
[x y] = uigetfile('*.jpg','Select an Image'); 
testimg = imread([y x]);
name=input('Enter New User:','s');
testimg=imresize(testimg,[255 255]);
figure,imshow(testimg); title('Input Eye Image'); 
axis off
%% Conversion
if size(testimg,3) == 3
      I = rgb2gray(testimg);
end
figure,imshow(I);title('Gray Image ');
impixelinfo;

R = testimg(:,:,1);
G = testimg(:,:,2); 
B = testimg(:,:,3);

%% otsu thresholding

  [M,N]=size(G);
  s=zeros(M,N);
  for i=1:M
      for j=1:N
          if G(i,j)<150;
             G(i,j)=0;
             
          end
      end
  end
  
figure,imshow(G),title('segmented Sclera'); 
enhances_contrast=adapthisteq(G);
enhances_intensity=imadjust(enhances_contrast);
binary_conv = imcomplement (enhances_intensity);
figure,imshow(enhances_intensity),title('Intensity enhanced');
figure,imshow(binary_conv),title('convoluted image');





%% Segmentation of Sclera vein

G=imcrop(G,[18 87 103 101]);
G=imresize(G,[255 255]);

sa = 0.1;
rt = mim(G,sa);

[tt1,e1,cmtx] = myThreshold(rt);

ms = 2;    
mk = msk(G,ms);

rt2 = 255*ones(M,N);
for i=1:M
    for j=1:N
        if rt(i,j)>=tt1 & mk(i,j)==255
            rt2(i,j)=0;
        end
    end
end
J = im2bw(rt2); 

J= ~J;
[Label,Num] = bwlabel(J);
Lmtx = zeros(Num+1,1);
for i=1:M
    for j=1:N
        Lmtx(double(Label(i,j))+1) = Lmtx(double(Label(i,j))+1) + 1;
    end
end
sLmtx = sort(Lmtx);
cp = 0.1;
for i=1:M
    for j=1:N
        if (Lmtx(double(Label(i,j)+1)) > cp) & (Lmtx(double(Label(i,j)+1)) ~= sLmtx(Num+1,1))
            J(i,j) = 0;
        else
            J(i,j) = 1;
        end
    end
end
for i=1:M
    for j=1:N
        if mk(i,j)==0
            J(i,j)=1;
        end
    end
end
figure; imshow(J,[]),title('Segmented Vein Pattern');

segv=J;
%% Gabour
Sx = 5; % Variances along x
Sy =5; % Variances along y
U = 5; % Centre frequencies  along x
V = 5; % Centre frequencies  along y
if isa(G,'double')~=1 
    segsclera = double(J);%convert to double type
end
for x = -fix(Sx):fix(Sx) %along x
    for y = -fix(Sy):fix(Sy) %along y
        G1(fix(Sx)+x+1,fix(Sy)+y+1) = (1/(2*pi*Sx*Sy))*exp(-.5*((x/Sx)^2+(y/Sy)^2)+2*pi*1i*(U*x+V*y));%filter eqn
    end
end
Imgabout = conv2(segsclera,double(imag(G1)),'same');%imaginary part
Regabout = conv2(segsclera,double(real(G1)),'same');%real part
gabout= sqrt(Imgabout.*Imgabout + Regabout.*Regabout);%final gabor filter,real + imaginary
figure,imshow(gabout,[]);title('Gabor Features');



%% Morphological
SE = strel('rectangle',[40 30]);
BW2 = imerode(J,SE);
figure,imshow(BW2),title('Eroded Image');

BW3 = imdilate(BW2,SE);
figure,imshow(BW3),title('Dialted Image');

%% feature extraction

points = detectSURFFeatures(J);
figure,imshow(J); hold on;
plot(points.selectStrongest(30));


%% Template:

sclera_temp = sclera_template(J);
[row1 col1] = size(sclera_temp);



%% Key Image
key=imresize(sclera_temp,[16 16]);
key=im2bw(key,0.3);
key_value=key;
figure,imshow(key),title('Key Image');


%% Save/Train New Database Enrollment

data_test=(key_value); %% rename difrnt data names for each database

% save the features

% save test8 data_test; %% to store values inside name of database(Create and rename the Db name)
save(['Database/' name '.mat'],'data_test'); %Mat encrypt
wait();

msgbox('Database Created Successfully');
