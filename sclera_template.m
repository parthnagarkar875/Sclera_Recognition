function template = iris1(I)  
% ----------------------------------------------------------------------- %
I = imresize(I,[256 256]);
[row col dim] = size(I);

if dim == 3
    I = rgb2gray(I);
end

% figure,imshow(I);
% title('Eye Image');
I1 = hist_eq(I);  % Histogram Equalization
% figure,imshow(uint8(I1));
% title('Histogram Equalization');
template = createiristemplate(I1);

figure,imshow(template);
title('Sclera Template');





function I1 = hist_eq(I) % Histogram  Equalization
I=double(I');
seq=double(I(:));
seq=sort(seq);
seq=round(seq);
value=[];
count=[];
while ~isempty(seq)
    pos=find(seq==seq(1));
    c=length(pos);
    value=[value seq(1)];
    count=[count c];
    seq(pos)=[];
end
cdf=[];
for i=1:length(value)
    cdf=[cdf sum(count(1:i))];
end
cdf_min=min(cdf);
[M N]=size(I);
h=zeros(1,length(value));
for v=1:length(value)
    h(v)=round(((cdf(v)-cdf_min)/((M*N)-cdf_min)).*255);
end
I1=zeros(M,N);
S=sort(value);
for i=1:length(value)
    pos=(I==S(i));
    I1(pos)=h(i);
end
I1=I1';


function  [template mask] = createiristemplate(eyeimage)

% createiristemplate - generates a biometric template from an iris in
% an eye image.
%
% Usage: 
% [template, mask] = createiristemplate(eyeimage_filename)
% Arguments:
%	eyeimage_filename   - the file name of the eye image
%
% Output:
%	template		    - the binary iris biometric template
%	mask			    - the binary iris noise mask

irisnum = 1;

%normalisation parameters
radial_res = 20;
angular_res = 240;
% with these settings a 9600 bit iris template is
% created

%feature encoding parameters
nscales = 1;
minWaveLength = 18;
mult = 1;                 % not applicable if using nscales = 1
sigmaOnf = 0.5;
 
[circleiris circlepupil imagewithnoise] = segmentiris(eyeimage);

% perform normalisation

[polar_array noise_array] = normaliseiris(imagewithnoise, circleiris(2),...
    circleiris(1), circleiris(3), circlepupil(2), ...
    circlepupil(1), circlepupil(3),irisnum, radial_res, angular_res);


% perform feature encoding
[template mask] = encode(polar_array, noise_array,...
    nscales, minWaveLength, mult, sigmaOnf);

% figure,imshow(template);
% xlabel('Template');
% set(1, 'pos', [190 392 644 128]);

% figure,imshow(mask);
% xlabel('Mask');
% set(2, 'pos', [190 180 644 128]);

function [circleiris, circlepupil, imagewithnoise] = segmentiris(eyeimage)

% segmentiris - peforms automatic segmentation of the iris region
% from an eye image. Also isolates noise areas such as occluding
% eyelids and eyelashes.
%
% Usage: 
% [circleiris, circlepupil, imagewithnoise] = segmentiris(image)
%
% Arguments:
%	eyeimage		- the input eye image
%	
% Output:
%	circleiris	    - centre coordinates and radius
%			          of the detected iris boundary
%	circlepupil	    - centre coordinates and radius
%			          of the detected pupil boundary
%	imagewithnoise	- original eye image, but with
%			          location of noise marked with
%			          NaN values
%

% define range of pupil & iris radii

%CASIA
lpupilradius = 28;
upupilradius = 75;
lirisradius = 80;
uirisradius = 150;

%    %LIONS
%    lpupilradius = 32;
%    upupilradius = 85;
%    lirisradius = 145;
%    uirisradius = 169;


% define scaling factor to speed up Hough transform
scaling = 0.4;



% find the iris boundary
[row, col, r] = findcircle(eyeimage, lirisradius, ...
    uirisradius, scaling, 2, 0.20, 0.19, 1.00, 0.00);

circleiris = [row col r];

rowd = double(row);
cold = double(col);
rd = double(r);

irl = round(rowd-rd);
iru = round(rowd+rd);
icl = round(cold-rd);
icu = round(cold+rd);

imgsize = size(eyeimage);

if irl < 1 
    irl = 1;
end

if icl < 1
    icl = 1;
end

if iru > imgsize(1)
    iru = imgsize(1);
end

if icu > imgsize(2)
    icu = imgsize(2);
end

% to find the inner pupil, use just the region within the previously
% detected iris boundary
imagepupil = eyeimage( irl:iru,icl:icu);

%find pupil boundary
[rowp, colp, r] = findcircle(imagepupil, lpupilradius, ... 
    upupilradius ,0.6,2,0.25,0.25,1.00,1.00);

rowp = double(rowp);
colp = double(colp);
r = double(r);

row = double(irl) + rowp;
col = double(icl) + colp;

row = round(row);
col = round(col);

circlepupil = [row col r];

% set up array for recording noise regions
% noise pixels will have NaN values
imagewithnoise = double(eyeimage);

%find top eyelid
topeyelid = imagepupil(1:(rowp-r), :);
lines = findline(topeyelid);

if size(lines,1) > 0
    [xl yl] = linecoords(lines, size(topeyelid));
    yl = double(yl) + irl-1;
    xl = double(xl) + icl-1;
    
    yla = max(yl);
    
    y2 = 1:yla;
    
    ind3 = sub2ind(size(eyeimage), yl, xl);
    imagewithnoise(ind3) = NaN;
    
    imagewithnoise(y2, xl) = NaN;
end

%find bottom eyelid
bottomeyelid = imagepupil((rowp+r):size(imagepupil,1),:);
lines = findline(bottomeyelid);

if size(lines,1) > 0
    
    [xl yl] = linecoords(lines, size(bottomeyelid));
    yl = double(yl)+ irl+rowp+r-2;
    xl = double(xl) + icl-1;
    
    yla = min(yl);
    
    y2 = yla:size(eyeimage, 1);
    
    ind4 = sub2ind(size(eyeimage), yl, xl);
    imagewithnoise(ind4) = NaN;
    imagewithnoise(y2, xl) = NaN;
    
end

%For CASIA, eliminate eyelashes by thresholding
ref = eyeimage < 100;
coords = find(ref==1);
imagewithnoise(coords) = NaN;

function [row, col, r] = findcircle(image,lradius,uradius,scaling, ...
    sigma, hithres, lowthres, vert, horz)


% findcircle - returns the coordinates of a circle in an image using the Hough transform
% and Canny edge detection to create the edge map.
%
% Usage: 
% [row, col, r] = findcircle(image,lradius,uradius,scaling, sigma, hithres, lowthres, vert, horz)
%
% Arguments:
%	image		    - the image in which to find circles
%	lradius		    - lower radius to search for
%	uradius		    - upper radius to search for
%	scaling		    - scaling factor for speeding up the
%			          Hough transform
%	sigma		    - amount of Gaussian smoothing to
%			          apply for creating edge map.
%	hithres		    - threshold for creating edge map
%	lowthres	    - threshold for connected edges
%	vert		    - vertical edge contribution (0-1)
%	horz		    - horizontal edge contribution (0-1)
%	
% Output:
%	circleiris	    - centre coordinates and radius
%			          of the detected iris boundary
%	circlepupil	    - centre coordinates and radius
%			          of the detected pupil boundary
%	imagewithnoise	- original eye image, but with
%			          location of noise marked with
%			          NaN values
%


lradsc = round(lradius*scaling);
uradsc = round(uradius*scaling);
rd = round(uradius*scaling - lradius*scaling);

% generate the edge image
[I2 or] = canny(image, sigma, scaling, vert, horz);
I3 = adjgamma(I2, 1.9);
I4 = nonmaxsup(I3, or, 1.5);
edgeimage = hysthresh(I4, hithres, lowthres);

% perform the circular Hough transform
h = houghcircle(edgeimage, lradsc, uradsc);

maxtotal = 0;

% find the maximum in the Hough space, and hence
% the parameters of the circle
for i=1:rd
    
    layer = h(:,:,i);
    [maxlayer] = max(max(layer));
    
    
    if maxlayer > maxtotal
        
        maxtotal = maxlayer;
        
        
        r = int32((lradsc+i) / scaling);
        
        [row,col] = ( find(layer == maxlayer) );
        
        
        row = int32(row(1) / scaling); % returns only first max value
        col = int32(col(1) / scaling);    
        
    end   
    
end

function [gradient, or] = canny(im, sigma, scaling, vert, horz)


% CANNY - Canny edge detection
%
% Function to perform Canny edge detection. 
% 
%
% Usage: [gradient or] = canny(im, sigma)
%
% Arguments:   im       - image to be procesed
%              sigma    - standard deviation of Gaussian smoothing filter
%                      (typically 1)
%		       scaling  - factor to reduce input image by
%		       vert     - weighting for vertical gradients
%		       horz     - weighting for horizontal gradients
%
% Returns:     gradient - edge strength image (gradient amplitude)
%              or       - orientation image (in degrees 0-180, positive
%                         anti-clockwise)
%
% See also:  NONMAXSUP, HYSTHRESH
% 
%


xscaling = vert;
yscaling = horz;

hsize = [6*sigma+1, 6*sigma+1];   % The filter size.

gaussian = fspecial('gaussian',hsize,sigma);
im = filter2(gaussian,im);        % Smoothed image.

im = imresize(im, scaling);

[rows, cols] = size(im);

h =  [  im(:,2:cols)  zeros(rows,1) ] - [  zeros(rows,1)  im(:,1:cols-1)  ];
v =  [  im(2:rows,:); zeros(1,cols) ] - [  zeros(1,cols); im(1:rows-1,:)  ];
d1 = [  im(2:rows,2:cols) zeros(rows-1,1); zeros(1,cols) ] - ...
                               [ zeros(1,cols); zeros(rows-1,1) im(1:rows-1,1:cols-1)  ];
d2 = [  zeros(1,cols); im(1:rows-1,2:cols) zeros(rows-1,1);  ] - ...
                               [ zeros(rows-1,1) im(2:rows,1:cols-1); zeros(1,cols)   ];

X = ( h + (d1 + d2)/2.0 ) * xscaling;
Y = ( v + (d1 - d2)/2.0 ) * yscaling;

gradient = sqrt(X.*X + Y.*Y); % Gradient amplitude.
or = atan2(-Y, X);            % Angles -pi to + pi.
neg = or<0;                   % Map angles to 0-pi.
or = or.*~neg + (or+pi).*neg; 
 % Convert to degrees.
or = or*180/pi;     

function newim = adjgamma(im, g)


% ADJGAMMA - Adjusts image gamma.
%
% function g = adjgamma(im, g)
%
% Arguments:
%            im     - image to be processed.
%            g      - image gamma value.
%                     Values in the range 0-1 enhance contrast of bright
%                     regions, values > 1 enhance contrast in dark
%                     regions. 



    if g <= 0
	error('Gamma value must be > 0');
    end

    if isa(im,'uint8');
	newim = double(im);
    else 
	newim = im;
    end
    	
    % rescale range 0-1
    newim = newim-min(min(newim));
    newim = newim./max(max(newim));
    
    % Apply gamma function
    newim =  newim.^(1/g);   
    
    
function im = nonmaxsup(inimage, orient, radius)


% NONMAXSUP
%
% Usage:
%          im = nonmaxsup(inimage, orient, radius);
%
% Function for performing non-maxima suppression on an image using an
% orientation image.  It is assumed that the orientation image gives 
% feature normal orientation angles in degrees (0-180).
%
% input:
%   inimage - image to be non-maxima suppressed.
% 
%   orient  - image containing feature normal orientation angles in degrees
%             (0-180), angles positive anti-clockwise.
% 
%   radius  - distance in pixel units to be looked at on each side of each
%             pixel when determining whether it is a local maxima or not.
%             (Suggested value about 1.2 - 1.5)
%
% Note: This function is slow (1 - 2 mins to process a 256x256 image).  It uses
% bilinear interpolation to estimate intensity values at ideal, real-valued pixel 
% locations on each side of pixels to determine if they are local maxima.
%


if size(inimage) ~= size(orient)
  error('image and orientation image are of different sizes');
end

if radius < 1
  error('radius must be >= 1');
end

[rows,cols] = size(inimage);
im = zeros(rows,cols);        % Preallocate memory for output image for speed
iradius = ceil(radius);

% Precalculate x and y offsets relative to centre pixel for each orientation angle 

angle = [0:180].*pi/180;    % Array of angles in 1 degree increments (but in radians).
xoff = radius*cos(angle);   % x and y offset of points at specified radius and angle
yoff = radius*sin(angle);   % from each reference position.

hfrac = xoff - floor(xoff); % Fractional offset of xoff relative to integer location
vfrac = yoff - floor(yoff); % Fractional offset of yoff relative to integer location

orient = fix(orient)+1;     % Orientations start at 0 degrees but arrays start
                            % with index 1.

% Now run through the image interpolating grey values on each side
% of the centre pixel to be used for the non-maximal suppression.

for row = (iradius+1):(rows - iradius)
  for col = (iradius+1):(cols - iradius) 

    or = orient(row,col);   % Index into precomputed arrays

    x = col + xoff(or);     % x, y location on one side of the point in question
    y = row - yoff(or);

    fx = floor(x);          % Get integer pixel locations that surround location x,y
    cx = ceil(x);
    fy = floor(y);
    cy = ceil(y);
    tl = inimage(fy,fx);    % Value at top left integer pixel location.
    tr = inimage(fy,cx);    % top right
    bl = inimage(cy,fx);    % bottom left
    br = inimage(cy,cx);    % bottom right

    upperavg = tl + hfrac(or) * (tr - tl);  % Now use bilinear interpolation to
    loweravg = bl + hfrac(or) * (br - bl);  % estimate value at x,y
    v1 = upperavg + vfrac(or) * (loweravg - upperavg);

  if inimage(row, col) > v1 % We need to check the value on the other side...

    x = col - xoff(or);     % x, y location on the `other side' of the point in question
    y = row + yoff(or);

    fx = floor(x);
    cx = ceil(x);
    fy = floor(y);
    cy = ceil(y);
    tl = inimage(fy,fx);    % Value at top left integer pixel location.
    tr = inimage(fy,cx);    % top right
    bl = inimage(cy,fx);    % bottom left
    br = inimage(cy,cx);    % bottom right
    upperavg = tl + hfrac(or) * (tr - tl);
    loweravg = bl + hfrac(or) * (br - bl);
    v2 = upperavg + vfrac(or) * (loweravg - upperavg);

    if inimage(row,col) > v2            % This is a local maximum.
      im(row, col) = inimage(row, col); % Record value in the output image.
    end

   end
  end
end


function bw = hysthresh(im, T1, T2)


% HYSTHRESH - Hysteresis thresholding
%
% Usage: bw = hysthresh(im, T1, T2)
%
% Arguments:
%             im  - image to be thresholded (assumed to be non-negative)
%             T1  - upper threshold value
%             T2  - lower threshold value
%
% Returns:
%             bw  - the thresholded image (containing values 0 or 1)
%
% Function performs hysteresis thresholding of an image.
% All pixels with values above threshold T1 are marked as edges
% All pixels that are adjacent to points that have been marked as edges
% and with values above threshold T2 are also marked as edges. Eight
% connectivity is used.
%
% It is assumed that the input image is non-negative
%
% Peter Kovesi          December 1996  - Original version
%                       March    2001  - Speed improvements made (~4x)
% 
%
% A stack (implemented as an array) is used to keep track of all the
% indices of pixels that need to be checked.
% Note: For speed the number of conditional tests have been minimised
% This results in the top and bottom edges of the image being considered to
% be connected.  This may cause some stray edges to be propagated further than 
% they should be from the top or bottom.
%


if (T2 > T1 | T2 < 0 | T1 < 0)  % Check thesholds are sensible
  error('T1 must be >= T2 and both must be >= 0 ');
end

[rows, cols] = size(im);    % Precompute some values for speed and convenience.
rc = rows*cols;
rcmr = rc - rows;
rp1 = rows+1;

bw = im(:);                 % Make image into a column vector
pix = find(bw > T1);        % Find indices of all pixels with value > T1
npix = size(pix,1);         % Find the number of pixels with value > T1

stack = zeros(rows*cols,1); % Create a stack array (that should never
                            % overflow!)

stack(1:npix) = pix;        % Put all the edge points on the stack
stp = npix;                 % set stack pointer
for k = 1:npix
    bw(pix(k)) = -1;        % mark points as edges
end


% Precompute an array, O, of index offset values that correspond to the eight 
% surrounding pixels of any point. Note that the image was transformed into
% a column vector, so if we reshape the image back to a square the indices 
% surrounding a pixel with index, n, will be:
%              n-rows-1   n-1   n+rows-1
%
%               n-rows     n     n+rows
%                     
%              n-rows+1   n+1   n+rows+1

O = [-1, 1, -rows-1, -rows, -rows+1, rows-1, rows, rows+1];

while stp ~= 0            % While the stack is not empty
    v = stack(stp);         % Pop next index off the stack
    stp = stp - 1;
    
    if v > rp1 & v < rcmr   % Prevent us from generating illegal indices
			    % Now look at surrounding pixels to see if they
                            % should be pushed onto the stack to be
                            % processed as well.
       index = O+v;	    % Calculate indices of points around this pixel.	    
       for l = 1:8
           ind = index(l);
           if bw(ind) > T2   % if value > T2,
               stp = stp+1;  % push index onto the stack.
               stack(stp) = ind;
               bw(ind) = -1; % mark this as an edge point
           end
       end
    end
end



bw = (bw == -1);            % Finally zero out anything that was not an edge 
% and reshape the image
bw = reshape(bw,rows,cols); 


function h = houghcircle(edgeim, rmin, rmax)


% houghcircle - takes an edge map image, and performs the Hough transform
% for finding circles in the image.
%
% Usage: 
% h = houghcircle(edgeim, rmin, rmax)
%
% Arguments:
%	edgeim      - the edge map image to be transformed
%   rmin, rmax  - the minimum and maximum radius values
%                 of circles to search for
% Output:
%	h           - the Hough transform
%


[rows,cols] = size(edgeim);
nradii = rmax-rmin+1;
h = zeros(rows,cols,nradii);

[y,x] = find(edgeim~=0);

%for each edge point, draw circles of different radii
for index=1:size(y)
    
    cx = x(index);
    cy = y(index);
    
    for n=1:nradii
        
        h(:,:,n) = addcircle(h(:,:,n),[cx,cy],n+rmin);
        
    end
    
end


function lines = findline(image)


% findline - returns the coordinates of a line in an image using the
% linear Hough transform and Canny edge detection to create
% the edge map.
%
% Usage: 
% lines = findline(image)
%
% Arguments:
%	image   - the input image
%
% Output:
%	lines   - parameters of the detected line in polar form
%


[I2 or] = canny(image, 2, 1, 0.00, 1.00);

I3 = adjgamma(I2, 1.9);
I4 = nonmaxsup(I3, or, 1.5);
edgeimage = hysthresh(I4, 0.20, 0.15);


theta = (0:179)';
[R, xp] = radon(edgeimage, theta);

maxv = max(max(R));

if maxv > 25
    i = find(R == max(max(R)));
else
    lines = [];
    return;
end

[foo, ind] = sort(-R(i));
u = size(i,1);
k = i(ind(1:u));
[y,x]=ind2sub(size(R),k);
t = -theta(x)*pi/180;
r = xp(y);

lines = [cos(t) sin(t) -r];

cx = size(image,2)/2-1;
cy = size(image,1)/2-1;
lines(:,3) = lines(:,3) - lines(:,1)*cx - lines(:,2)*cy;



function [polar_array, polar_noise] = normaliseiris(image, x_iris, y_iris, r_iris,...
x_pupil, y_pupil, r_pupil,eyeimage_filename, radpixels, angulardiv)


% normaliseiris - performs normalisation of the iris region by
% unwraping the circular region into a rectangular block of
% constant dimensions.
%
% Usage: 
% [polar_array, polar_noise] = normaliseiris(image, x_iris, y_iris, r_iris,...
% x_pupil, y_pupil, r_pupil,eyeimage_filename, radpixels, angulardiv)
%
% Arguments:
% image                 - the input eye image to extract iris data from
% x_iris                - the x coordinate of the circle defining the iris
%                         boundary
% y_iris                - the y coordinate of the circle defining the iris
%                         boundary
% r_iris                - the radius of the circle defining the iris
%                         boundary
% x_pupil               - the x coordinate of the circle defining the pupil
%                         boundary
% y_pupil               - the y coordinate of the circle defining the pupil
%                         boundary
% r_pupil               - the radius of the circle defining the pupil
%                         boundary
% eyeimage_filename     - original filename of the input eye image
% radpixels             - radial resolution, defines vertical dimension of
%                         normalised representation
% angulardiv            - angular resolution, defines horizontal dimension
%                         of normalised representation
%
% Output:
% polar_array
% polar_noise




radiuspixels = radpixels + 2;
angledivisions = angulardiv-1;



theta = 0:2*pi/angledivisions:2*pi;

x_iris = double(x_iris);
y_iris = double(y_iris);
r_iris = double(r_iris);

x_pupil = double(x_pupil);
y_pupil = double(y_pupil);
r_pupil = double(r_pupil);

% calculate displacement of pupil center from the iris center
ox = x_pupil - x_iris;
oy = y_pupil - y_iris;

if ox <= 0
    sgn = -1;
elseif ox > 0
    sgn = 1;
end

if ox==0 && oy > 0
    
    sgn = 1;
    
end


theta = double(theta);

a = ones(1,angledivisions+1)* (ox^2 + oy^2);

% need to do something for ox = 0
if ox == 0
    phi = pi/2;
else
    phi = atan(oy/ox);
end

b = sgn.*cos(pi - phi - theta);

% calculate radius around the iris as a function of the angle
r = (sqrt(a).*b) + ( sqrt( a.*(b.^2) - (a - (r_iris^2))));

r = r - r_pupil;

rmat = ones(1,radiuspixels)'*r;

rmat = rmat.* (ones(angledivisions+1,1)*[0:1/(radiuspixels-1):1])';
rmat = rmat + r_pupil;


% exclude values at the boundary of the pupil iris border, and the iris scelra border
% as these may not correspond to areas in the iris region and will introduce noise.
%
% ie don't take the outside rings as iris data.
rmat  = rmat(2:(radiuspixels-1), :);

% calculate cartesian location of each data point around the circular iris
% region
xcosmat = ones(radiuspixels-2,1)*cos(theta);
xsinmat = ones(radiuspixels-2,1)*sin(theta);

xo = rmat.*xcosmat;    
yo = rmat.*xsinmat;

xo = x_pupil+xo;
yo = y_pupil-yo;

% extract intensity values into the normalised polar representation through
% interpolation
[x,y] = meshgrid(1:size(image,2),1:size(image,1));  
polar_array = interp2(x,y,image,xo,yo);

% create noise array with location of NaNs in polar_array
polar_noise = zeros(size(polar_array));
coords = find(isnan(polar_array));
polar_noise(coords) = 1;

polar_array = double(polar_array)./255;



%replace NaNs before performing feature encoding
coords = find(isnan(polar_array));
polar_array2 = polar_array;
polar_array2(coords) = 0.5;
avg = sum(sum(polar_array2)) / (size(polar_array,1)*size(polar_array,2));
polar_array(coords) = avg;



function [EO, filtersum] = gaborconvolve(im, nscale, minWaveLength, mult, ...
    sigmaOnf)


% gaborconvolve - function for convolving each row of an image with 1D log-Gabor filters
%
% Usage: 
% [template, mask] = createiristemplate(eyeimage_filename)
%
% Arguments:
%   im              - the image to convolve
%   nscale          - number of filters to use
%   minWaveLength   - wavelength of the basis filter
%   mult            - multiplicative factor between each filter
%   sigmaOnf        - Ratio of the standard deviation of the Gaussian describing
%                     the log Gabor filter's transfer function in the frequency
%                     domain to the filter center frequency.
%
% Output:
%   E0              - a 1D cell array of complex valued comvolution results
%


[rows cols] = size(im);		
filtersum = zeros(1,size(im,2));

EO = cell(1, nscale);          % Pre-allocate cell array

ndata = cols;
if mod(ndata,2) == 1             % If there is an odd No of data points 
    ndata = ndata-1;               % throw away the last one.
end

logGabor  = zeros(1,ndata);
result = zeros(rows,ndata);

radius =  [0:fix(ndata/2)]/fix(ndata/2)/2;  % Frequency values 0 - 0.5
radius(1) = 1;

wavelength = minWaveLength;        % Initialize filter wavelength.


for s = 1:nscale,                  % For each scale.  
    
    % Construct the filter - first calculate the radial filter component.
    fo = 1.0/wavelength;                  % Centre frequency of filter.
    rfo = fo/0.5;                         % Normalised radius from centre of frequency plane 
    % corresponding to fo.
    logGabor(1:ndata/2+1) = exp((-(log(radius/fo)).^2) / (2 * log(sigmaOnf)^2));  
    logGabor(1) = 0;  
    
    filter = logGabor;
    
    filtersum = filtersum+filter;
    
    % for each row of the input image, do the convolution, back transform
    for r = 1:rows	% For each row
        
        signal = im(r,1:ndata);
        
        
        imagefft = fft( signal );
        
        
        result(r,:) = ifft(imagefft .* filter);
        
    end    
    % save the ouput for each scale
    EO{s} = result;
    
    wavelength = wavelength * mult;       % Finally calculate Wavelength of next filter
end                                     % ... and process the next scale

filtersum = fftshift(filtersum);



function [template, mask] = encode(polar_array,noise_array, nscales, minWaveLength, mult, sigmaOnf)

% encode - generates a biometric template from the normalised iris region,
% also generates corresponding noise mask
%
% Usage: 
% [template, mask] = encode(polar_array,noise_array, nscales,...
% minWaveLength, mult, sigmaOnf)
%
% Arguments:
% polar_array       - normalised iris region
% noise_array       - corresponding normalised noise region map
% nscales           - number of filters to use in encoding
% minWaveLength     - base wavelength
% mult              - multicative factor between each filter
% sigmaOnf          - bandwidth parameter
%
% Output:
% template          - the binary iris biometric template
% mask              - the binary iris noise mask
%


% convolve normalised region with Gabor filters
[E0 filtersum] = gaborconvolve(polar_array, nscales, minWaveLength, mult, sigmaOnf);

length = size(polar_array,2)*2*nscales;

template = zeros(size(polar_array,1), length);

length2 = size(polar_array,2);
h = 1:size(polar_array,1);

%create the iris template

mask = zeros(size(template));

for k=1:nscales
    
    E1 = E0{k};
    
    %Phase quantisation
    H1 = real(E1) > 0;
    H2 = imag(E1) > 0;
    
    % if amplitude is close to zero then
    % phase data is not useful, so mark off
    % in the noise mask
    H3 = abs(E1) < 0.0001;
    
    
    for i=0:(length2-1)
                
        ja = double(2*nscales*(i));
        %construct the biometric template
        template(h,ja+(2*k)-1) = H1(h, i+1);
        template(h,ja+(2*k)) = H2(h,i+1);
        
        %create noise mask
        mask(h,ja+(2*k)-1) = noise_array(h, i+1) | H3(h, i+1);
        mask(h,ja+(2*k)) =   noise_array(h, i+1) | H3(h, i+1);
        
    end
    
end 


function h = addcircle(h, c, radius, weight)



% ADDCIRCLE
%
% A circle generator for adding (drawing) weights into a Hough accumumator
% array.
%
% Usage:  h = addcircle(h, c, radius, weight)
% 
% Arguments:
%            h      - 2D accumulator array.
%            c      - [x,y] coords of centre of circle.
%            radius - radius of the circle
%            weight - optional weight of values to be added to the
%                     accumulator array (defaults to 1)
%
% Returns:   h - Updated accumulator array.


    [hr, hc] = size(h);
    
    if nargin == 3
	weight = 1;
    end
    
    % c and radius must be integers
    if any(c-fix(c))
	error('Circle centre must be in integer coordinates');
    end
    
    if radius-fix(radius)
	error('Radius must be an integer');
    end
    
    x = 0:fix(radius/sqrt(2));
    costheta = sqrt(1 - (x.^2 / radius^2));
    y = round(radius*costheta);
    
    % Now fill in the 8-way symmetric points on a circle given coords 
    % [px py] of a point on the circle.
    
    px = c(2) + [x  y  y  x -x -y -y -x];
    py = c(1) + [y  x -x -y -y -x  x  y];

    % Cull points that are outside limits
    validx = px>=1 & px<=hr;
    validy = py>=1 & py<=hc;    
    valid = find(validx & validy);

    px = px(valid);
    py = py(valid);
    
    ind = px+(py-1)*hr;
    h(ind) = h(ind) + weight;

    
    
function [x,y] = linecoords(lines, imsize)


% linecoords - returns the x y coordinates of positions along a line
%
% Usage: 
% [x,y] = linecoords(lines, imsize)
%
% Arguments:
%	lines       - an array containing parameters of the line in
%                 form
%   imsize      - size of the image, needed so that x y coordinates
%                 are within the image boundary
%
% Output:
%	x           - x coordinates
%	y           - corresponding y coordinates
%


xd = [1:imsize(2)];
yd = (-lines(3) - lines(1)*xd ) / lines(2);

coords = find(yd>imsize(1));
yd(coords) = imsize(1);
coords = find(yd<1);
yd(coords) = 1;

x = int32(xd);
y = int32(yd);   




