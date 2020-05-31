function FDI=siftfeat(jk)
I=jk;                  % to read the image
I_read=imresize(I,[256 256]);
I_enlarge=imresize(I_read,[512 512]);       % image resize
% I=rgb2gray(I_enlarge);                      % rgb to gray conversion
I=I_enlarge;
imshow(I);
I=im2double(I);

I_temp=I;
original=I;
octave1=[];                                 % initializing empty octave's matrix
octave2=[];
octave3=[];
%% Forming octaves
% in this section different octaves are formed using differnt sigma(scale) value
tic;
k2=0;                                       % first octave level
[m,n]=size(I);
I_temp(m:m+4,n:n+4)=0;                      % zero padding
clear c;
tic;
for k1=0:3                                  % for different sigma value (different level in one octave)                       
    k=sqrt(2);
sigma=(k^(k1+(2*k2)))*1.6;

for x=-2:2                                  % generating gaussian filter coefficent
    for y=-2:2
        h(x+3,y+3)=(1/((2*pi)*((k*sigma)*(k*sigma))))*exp(-((x*x)+(y*y))/(2*(k*k)*(sigma*sigma))); 
    end
end


for i=1:m
    for j=1:n
        temp1=I_temp(i:i+4,j:j+4)'.*h;      %convolving image with gaussian filter
        conv_1(i,j)=sum(sum(temp1));
    end
end
% figure, imshow(conv_1(1:m-4,1:n-4));
octave1=[octave1 conv_1];                   % store gaussian filtered image of different scale in octave1
end
%% octave 2
% second level of octave , staring with k^2 sigma
clear I_temp;
% k2=1;
I_temp2=imresize(original,1/((k2+1)*2));    % reduce image size by 2 in both x & y direction
 k2=1;
y=size(I_temp2)
[m,n]=size(I_temp2);
I_temp2(m:m+4,n:n+4)=0;
clear c;

for k1=0:3
    k=sqrt(2);
sigma=(k^(k1+(2*k2)))*1.6;

for x=-2:2
    for y=-2:2
        h1(x+3,y+3)=(1/((2*pi)*((k*sigma)*(k*sigma))))*exp(-((x*x)+(y*y))/(2*(k*k)*(sigma*sigma)));     % gaussian filter coefficents
    end
end

for i=1:m
    for j=1:n
        temp2=I_temp2(i:i+4,j:j+4)'.*h1;        % convolution of re-sized image with gaussian filter
        conv_2(i,j)=sum(sum(temp2));
    end
end
% figure, imshow(conv_2(1:m-4,1:n-4));
octave2=[octave2 conv_2];                       % store all second level image of different sigma in octave2
end
%% octave 3
% third level of octave starting with k^4 times sigma value
clear I_temp;
% k2=2;
I_temp3=imresize(original,1/((k2+1)*2));        % reduce orignal image by a factor of 4 in x and y
z=size(I_temp3)
 k2=2;
[m,n]=size(I_temp3);
I_temp3(m:m+4,n:n+4)=0;
clear c;

for k1=0:4
    k=sqrt(2);
sigma=(k^(k1+(2*k2)))*1.6;


for x=-2:2
    for y=-2:2
        h2(x+3,y+3)=(1/((2*pi)*((k*sigma)*(k*sigma))))*exp(-((x*x)+(y*y))/(2*(k*k)*(sigma*sigma))); %filter coefficent
    end
end


for i=1:m
    for j=1:n
        temp3=I_temp3(i:i+4,j:j+4)'.*h2;        %convolution of re-sized image with gaussian filter
        conv_3(i,j)=sum(sum(temp3));
    end
end
% figure, imshow(conv_3(1:m-4,1:n-4));
octave3=[octave3 conv_3];                       % store different level of image in octave3
end

fprintf('\n Time for Gaussian scale space construction: %.3f s\n',toc) ;

%% DOG---difference of gaussian
% in this section different level of gaussian filtered image are subtracted
% to form difference of gaussian image for further calculation
tic;

 diff_11=octave1(1:512,1:512)-octave1(1:512,513:1024);      % differnce of gaussian for octave1
 
 diff_12=octave1(1:512,513:1024)-octave1(1:512,1025:1536);
 
 diff_13=octave1(1:512,1025:1536)-octave1(1:512,1537:2048);
 
 
 diff_21=octave2(1:256,1:256)-octave2(1:256,257:512);        % difference of gaussian for octave2
 
 diff_22=octave2(1:256,257:512)-octave2(1:256,513:768);
 
 diff_23=octave2(1:256,513:768)-octave2(1:256,769:1024);
 

 diff_31=octave3(1:128,1:128)-octave3(1:128,129:256);        % difference of gaussian for octave3
 
diff_32=octave3(1:128,129:256)-octave3(1:128,257:384);

 diff_33=octave3(1:128,257:384)-octave3(1:128,385:512);
 
 fprintf('\n Time for Differential scale space construction: %.3f s\n',toc) ;

%% find exterma from DOG 
% in this scetion DOG image is used and extremum point is calculated
tic;
key=[];                             % empty matrix to store key points

x1=0;                               % local variables used to take if condition in account
y1=0;
z1=0;
f=0;

for i=2:511
    for j=2:511
%         x1=0;
% y1=0;
% z1=0;

        
if (((diff_12(i,j)>diff_12(i-1,j))&&(diff_12(i,j)>diff_12(i+1,j))....
        &&(diff_12(i,j)>diff_12(i,j-1))&&(diff_12(i,j)>diff_12(i+1,j+1))....
        &&(diff_12(i,j)>diff_12(i-1,j-1))&&(diff_12(i,j)>diff_12(i-1,j+1))....
        &&(diff_12(i,j)>diff_12(i+1,j-1))&&(diff_12(i,j)>diff_12(i,j+1))))
    x1=x1+1;
else
    continue;
end

if x1>0
    if((diff_12(i,j)>diff_13(i,j))&&(diff_12(i,j)>diff_13(i-1,j))....
            &&(diff_12(i,j)>diff_13(i+1,j))&&(diff_12(i,j)>diff_13(i,j-1))....
            &&(diff_12(i,j)>diff_13(i+1,j+1))&&(diff_12(i,j)>diff_13(i-1,j-1))....
            &&(diff_12(i,j)>diff_13(i-1,j+1))&&(diff_12(i,j)>diff_13(i+1,j-1))&&(diff_12(i,j)>diff_13(i,j+1)))
        y1=y1+1;
    else
        continue;
        
    end 
end
%   if y1>0
%        
%     if ((diff_12(i,j)>diff_11(i,j))&&(diff_12(i,j)>diff_11(i-1,j))&&(diff_12(i,j)>diff_11(i+1,j))&&(diff_12(i,j)>diff_11(i,j-1))&&(diff_12(i,j)>diff_11(i+1,j+1))&&(diff_12(i,j)>diff_11(i-1,j-1))&&(diff_12(i,j)>diff_11(i-1,j+1))&&(diff_12(i,j)>diff_11(i+1,j-1))&&(diff_12(i,j)>diff_11(i,j+1)))
%         z1=z1+1;
%     else 
%         continue;
%     end
%   end
  
  key(i,j)=diff_12(i,j);                    % store key point location if it is maximum in its neighbourhood on same scale and also on scale above and below
  f=1;
  

end
end
    
fprintf('\n Time for finding key points: %.3f s\n',toc) ;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
 
   if f==0
    x=0;
y=0;
z=0;

for i=2:511
    for j=2:511
        
if (((diff_12(i,j)<diff_12(i-1,j))&&(diff_12(i,j)<diff_12(i+1,j))....
        &&(diff_12(i,j)<diff_12(i,j-1))&&(diff_12(i,j)<diff_12(i+1,j+1))....
        &&(diff_12(i,j)<diff_12(i-1,j-1))&&(diff_12(i,j)<diff_12(i-1,j+1))....
        &&(diff_12(i,j)<diff_12(i+1,j-1))&&(diff_12(i,j)<diff_12(i,j+1))))
    x=x+1;
else
   continue;
end

if x>0
    if ((diff_12(i,j)<diff_13(i,j))&&(diff_12(i,j)<diff_13(i-1,j))....
            &&(diff_12(i,j)<diff_13(i+1,j))&&(diff_12(i,j)<diff_13(i,j-1))....
            &&(diff_12(i,j)<diff_13(i+1,j+1))&&(diff_12(i,j)<diff_13(i-1,j-1))....
            &&(diff_12(i,j)<diff_13(i-1,j+1))&&(diff_12(i,j)<diff_13(i+1,j-1))&&(diff_12(i,j)<diff_13(i,j+1)))
        y=y+1;
    else
        continue;
        
    end 
end
  if y>0
      
   if ((diff_12(i,j)<diff_11(i,j))&&(diff_12(i,j)<diff_11(i-1,j))....
           &&(diff_12(i,j)<diff_11(i+1,j))&&(diff_12(i,j)<diff_11(i,j-1))....
           &&(diff_12(i,j)<diff_11(i+1,j+1))&&(diff_12(i,j)<diff_11(i-1,j-1))....
           &&(diff_12(i,j)<diff_11(i-1,j+1))&&(diff_12(i,j)<diff_11(i+1,j-1))&&(diff_12(i,j)<diff_11(i,j+1)))
       z=z+1;
   else 
       continue;
   end
  end
  
  key(i,j)=diff_12(i,j);                % store key point location if it is minimum in its neighbourhood on same scale and also on scale above and below
  
  






end
    end
  
   end
  
 key1=key*255;
 figure,imshow(key1);                   % show image key points 
 
 %% finding key point location

[key_m,key_n]=size(key);
  r=1;  
key_p=[];                               % matrix to store key points locations
for i=1:key_m
    for j=1:key_n
    
    if key(i,j)>0
%         key_p=[key_p,i,j];
           key_p(r,1)=i;                
           key_p(r,2)=j;
           r=r+1;

    end
end
end
fprintf('\n toatl numer of key points extracted are : \n')
length(key_p)
 
%% magnitude and phase calculation

for i=2:511
    for j=2:511
        mag_1(i,j)=((diff_12(i+1,j)-diff_12(i-1,j))^2)+((diff_12(i,j+1)-diff_12(i,j-1))^2);
        phase(i,j)=atan2((diff_12(i,j+1)-diff_11(i,j-1)),(diff_12(i+1,j)-diff_11(i-1,j)));      % phase of all pixels
    end
end
mag=sqrt(mag_1);                % magnitude of all pixel of diff_12 level 

%% FINDING  ORIENTATION
% in this section first complete phase is distributed in 36 bin each of 10
% degree(pi/18) and phase of all the points in 5*5 neighbourhood of key
% point are stored in the bins and their magnitude is added and max
% magnitude among them will become orientation of key point and wtih
% direction and magnitude

for k=1:length(key_p)
    
m=key_p(k,1);               % extract key point and store in m & n
n=key_p(k,2);

if (m<=2)||(n<=2)||(m>=509)||(n>=509)
    continue;
end
temp_mag=mag(m-2:m+2,n-2:n+2);          % 5*5 neighborhood of magnitude @ (m,n)
temp_phase=phase(m-2:m+2,n-2:n+2);      % 5*5 neighborhood of phase @ (m,n)
clear bin_p;
clear bin_m;
s=1;
     
for i=1:5
    for j=1:5
        k1=1;
        for x=1:36                      % 36 bin each of 10 degree
            if temp_phase(i,j)>-pi+(k1-1)*0.1745 && temp_phase(i,j)<-pi+0.1745*k1
                
                bin_p(k1,s)=temp_phase(i,j);
                bin_m(k1,s)=temp_mag(i,j);
                s=s+1;
            end
            bin_p(k1,s)=0;              % bin for phase value
            bin_m(k1,s)=0;              % bin for magnitude value
            %s=s+1;
            k1=k1+1;
           
        end
    end
end

for i=1:36
d1=sum(bin_m(i,:));
magv(i,k)=d1;                       % magnitude of bins
end


end



max_mag=max(magv);              % maximum magnitude value

for i=1:length(key_p)
    
    if max_mag(1,i)==0
        continue;
    end
    
    j=find(magv(:,i)==(max_mag(1,i)));
    
    
    max_mag(2,i)=j;             % matrix that store max magnitude value and their corresponding bin to get orientation of key points
end

   %% DESCRIPTOR..
   %in this section discreptor vector of key points are found, for this a
   %16*16 neighbourhood around key point is taken and this space is divided
   %in 16 4*4 matrix and for each 4*4 matrix 8 directional hostogram is
   %used to get eight orientation and magnitude so for each key point we
   %have 16*8=128 feature discriptor
   
   tic; 
   for k=1:length(key_p)
    magv_2=[];
m=key_p(k,1);               % extract key points location
n=key_p(k,2);

 if (m<=8)||(n<=8)||(m>=503)||(n>=503)
     continue;
 end
temp_mag_d=mag(m-7:m+8,n-7:n+8);            % form 16*16 neifghbour around key point
temp_phase_d=phase(m-7:m+8,n-7:n+8);

store_phase=[];
for i=1:4:13
    for j=1:4:13
        vijj=temp_mag_d(i:i+3,j:j+3);
        vijj_phase=temp_phase_d(i:i+3,j:j+3);
        
    %end
    store_phase=[store_phase,vijj_phase];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    s=1;
    for i1=1:4                          % divide 16*16 region in 16 4*4 region
    for j1=1:4
        k1=1;
        for x=1:8                       % 8 bin each of 45 degree(pi/4)
            if vijj_phase(i1,j1)>-pi+(k1-1)*(pi/4) && vijj_phase(i1,j1)<-pi+(pi/4)*k1
                
                bin_p1(k1,s)=vijj_phase(i1,j1);
                bin_m1(k1,s)=vijj(i1,j1);
                s=s+1;
            end
            bin_p1(k1,s)=0;             % phase bin
            bin_m1(k1,s)=0;             % magnitude phase
            %s=s+1;
            k1=k1+1;
           
            
        end
    end
 end

for g=1:8
d2(:,1)=sum(bin_m1(g,:));
magv_2=[magv_2,d2];                 % magnitude of bins
end


    end


end

descriptor(k,:)=magv_2(:,:)';           % complete descriptor matrix for all key points.


    
   end
 fprintf('\n Time for calculating descriptor: %.3f s\n',toc) ;

