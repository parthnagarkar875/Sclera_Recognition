function varargout = mainguii(varargin)
% MAINGUII MATLAB code for mainguii.fig
%      MAINGUII, by itself, creates a new MAINGUII or raises the existing
%      singleton*.
%
%      H = MAINGUII returns the handle to a new MAINGUII or the handle to
%      the existing singleton*.
%
%      MAINGUII('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MAINGUII.M with the given input arguments.
%
%      MAINGUII('Property','Value',...) creates a new MAINGUII or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before mainguii_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to mainguii_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help mainguii

% Last Modified by GUIDE v2.5 19-Apr-2015 20:33:00

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @mainguii_OpeningFcn, ...
                   'gui_OutputFcn',  @mainguii_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before mainguii is made visible.
function mainguii_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to mainguii (see VARARGIN)

% Choose default command line output for mainguii
handles.output = hObject;
axes(handles.axes8);
imshow('3d.jpg');title('AMROKSHA Solutions','FontSize',15,'Color','b'); %(AMROKSHA :- AMey ROhit KSHipra Akshay)
guidata(hObject, handles);

% UIWAIT makes mainguii wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = mainguii_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global testimg

% Inputs

[x y] = uigetfile('*.jpg','Select an Image'); 
testimg = imread([y x]);
axes(handles.axes1);
testimg=imresize(testimg,[255 255]);
imshow(testimg); title('Input Eye Image','FontSize',15,'Color','r'); 
% axis off
% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global testimg
global I 

if size(testimg,3) == 3
      I = rgb2gray(testimg);
end
handles.axes1;
imshow(I);title('Pre Processed Image','FontSize',15,'Color','r'); 
impixelinfo;


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global testimg
global I 
global R
global G
global B 

R = testimg(:,:,1);
handles.axes1;imshow(R);title('Red plane Image','FontSize',15,'Color','r'); 
pause(1)
G = testimg(:,:,2); 
handles.axes1;
imshow(G);title('Green Processed Image','FontSize',15,'Color','r'); 
pause(1)
B = testimg(:,:,3);
handles.axes1;imshow(B);title('Blue plane Image','FontSize',15,'Color','r'); 
pause(1)


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global G
global enhances_contrast
global enhances_intensity
global M
global N

[M,N]=size(G);
  s=zeros(M,N);
  for i=1:M
      for j=1:N
          if G(i,j)<150;
             G(i,j)=0;
             
          end
      end
  end
  
handles.axes1;imshow(G),title('Segmented Sclera','FontSize',15,'Color','r'); 
enhances_contrast=adapthisteq(G);
enhances_intensity=imadjust(enhances_contrast);
binary_conv = imcomplement (enhances_intensity);
% figure,imshow(enhances_intensity),title('Intensity enhanced');
% figure,imshow(binary_conv),title('convoluted image');


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global G
global Gabor
global Hessian_Frangi
global Gaussian
v = get(handles.popupmenu1,'Value'); %get currently selected option from menu
if v == 2
   Sx = 5; % Variances along x
Sy =5; % Variances along y
U = 5; % Centre frequencies  along x
V = 5; % Centre frequencies  along y
if isa(G,'double')~=1 
    segsclera = double(G);%convert to double type
end
for x = -fix(Sx):fix(Sx) %along x
    for y = -fix(Sy):fix(Sy) %along y
        G1(fix(Sx)+x+1,fix(Sy)+y+1) = (1/(2*pi*Sx*Sy))*exp(-.5*((x/Sx)^2+(y/Sy)^2)+2*pi*1i*(U*x+V*y));%filter eqn
    end
end
Imgabout = conv2(segsclera,double(imag(G1)),'same');%imaginary part
Regabout = conv2(segsclera,double(real(G1)),'same');%real part
Gabor = sqrt(Imgabout.*Imgabout + Regabout.*Regabout);%final gabor filter,real + imaginary
handles.axes1;imshow(Gabor,[]);title('Gabor Features','FontSize',15,'Color','r');
elseif v == 1
    if isa(G,'double')~=1 
    segsclera = double(G);%convert to double type
end
    Hessian_Frangi=FrangiFilter2D(segsclera);
  handles.axes1;imshow(Hessian_Frangi,[]);title('Hessian Features','FontSize',15,'Color','r');
elseif v == 3
    if isa(G,'double')~=1 
    segsclera = double(G);%convert to double type
end
Gaussian= imgaussian(segsclera,2);    
handles.axes1;imshow(Gaussian,[]);title('Gaussian Features','FontSize',15,'Color','r');
end

% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global G
global J
global M
global N
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
handles.axes1; imshow(J,[]),title('Segmented Vein Pattern','FontSize',15,'Color','r');
imwrite(J,'veinseye.tif');
jk=imread('veinseye.tif');
segv=J;

% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global J
global jk
points = detectSURFFeatures(J);
handles.axes1;imshow(J); hold on;
plot(points.selectStrongest(100));
% %sift faetrue
% jk=im2double(jk);
% siftfeat(jk);



% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global J
global sclera_temp
global key
global key_value

sclera_temp = sclera_template(J);
[row1 col1] = size(sclera_temp);
key=imresize(sclera_temp,[16 16]);
key=im2bw(key,0.3);
key_value=key;
handles.axes1;imshow(key),title('Key Image');


% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(~, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global key_value

test=(key_value); 
wait2();

cd Database
 a=dir;count=0;
 dec=0;
 for i=1:size(a,1)
     if not(strcmp(a(i).name,'.')|strcmp(a(i).name,'..')|strcmp(a(i).name,'Thumbs.db'))
         count=count+1;
         temp=a(i).name;
         load(temp)
         corr_coeff=corr2(test,data_test);
         if corr_coeff>0.9
             name=temp(1:end-4);
             msgbox([name '--Authenticated Person' ]); 
             dec=1;
             break
         end
     end
 end
  cd ..
 if dec~=1
    errordlg('Person Authentication Failed','Database Error');
 end



% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1

val = get(hObject,'Value');
str = get(hObject,'String');
    
% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
