function varargout = SECURITY_SYSTEM_USING_SCLERA(varargin)
% SECURITY_SYSTEM_USING_SCLERA MATLAB code for SECURITY_SYSTEM_USING_SCLERA.fig
%      SECURITY_SYSTEM_USING_SCLERA, by itself, creates a new SECURITY_SYSTEM_USING_SCLERA or raises the existing
%      singleton*.
%
%      H = SECURITY_SYSTEM_USING_SCLERA returns the handle to a new SECURITY_SYSTEM_USING_SCLERA or the handle to
%      the existing singleton*.
%
%      SECURITY_SYSTEM_USING_SCLERA('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SECURITY_SYSTEM_USING_SCLERA.M with the given input arguments.
%
%      SECURITY_SYSTEM_USING_SCLERA('Property','Value',...) creates a new SECURITY_SYSTEM_USING_SCLERA or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SECURITY_SYSTEM_USING_SCLERA_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SECURITY_SYSTEM_USING_SCLERA_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SECURITY_SYSTEM_USING_SCLERA

% Last Modified by GUIDE v2.5 04-Feb-2018 09:32:27

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SECURITY_SYSTEM_USING_SCLERA_OpeningFcn, ...
                   'gui_OutputFcn',  @SECURITY_SYSTEM_USING_SCLERA_OutputFcn, ...
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


% --- Executes just before SECURITY_SYSTEM_USING_SCLERA is made visible.
function SECURITY_SYSTEM_USING_SCLERA_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SECURITY_SYSTEM_USING_SCLERA (see VARARGIN)

% Choose default command line output for SECURITY_SYSTEM_USING_SCLERA
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SECURITY_SYSTEM_USING_SCLERA wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SECURITY_SYSTEM_USING_SCLERA_OutputFcn(hObject, eventdata, handles) 
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

[x y] = uigetfile('*.jpg','Select an Image'); 
testimg = imread([y x]);
axes(handles.axes1);
testimg=imresize(testimg,[255 255]);
imshow(testimg); title('Input Eye Image','FontSize',15,'Color','r'); 


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global testimg
global I 
global R
global G
global B 

if size(testimg,3) == 3
      I = rgb2gray(testimg);
end

G = testimg(:,:,2); 
handles.axes1;imshow(G);title('Green Plane Extracted Image','FontSize',15,'Color','r'); 


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
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


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global G
global Gaussian


if isa(G,'double')~=1 
    segsclera = double(G);%convert to double type
end
Gaussian= imgaussian(segsclera,2);    
handles.axes1;imshow(Gaussian,[]);title('Gaussian Features','FontSize',15,'Color','r');


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
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


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global J
global jk
points = detectSURFFeatures(J);
handles.axes1;imshow(J); hold on;
plot(points.selectStrongest(100));


% % --- Executes on button press in pushbutton8.
% function pushbutton8_Callback(hObject, eventdata, handles)
% % hObject    handle to pushbutton8 (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% 


% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
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
