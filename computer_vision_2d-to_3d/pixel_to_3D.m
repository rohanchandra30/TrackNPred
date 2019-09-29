%% Convert 2D points to 3D world coordinates (Find the scale factor)

% Author: Rohan Chandra, copyright 2019

% Given the K, R, t matrices, we apply the inverse projection equation on
% image points. That is, [X,Y,Z,1] = {[K][R;t]}^{-1} X [u,v,1]. The image
% points, [u,v], correspond to interest points that could be corner points
% of certain vehicles whose real world dimensions are known. The goal is to
% approximately estimate the scale factor in the video.

%The steps are as follows:
% 1. Invert KRt. Store as a matrix.
% 2. Find key points manually and store as an array.
% 3. Apply inverse projection.
% 4. Map the pixel distances to real world distances to estimate the scale,
%    lambda


% OUTPUT: The final "vertical_mapping" tells us how many units 1 pixel of vertical
% distance corresponds to. We use units since we do not have depth
% information. In order to estimate an approximate scale, we provide the
% following information: An autorickshaw's vertical height is 1.7m. We can
% use this information to estimate an approximate units to meters scale.
% This code uses this example. In particular, p1 and p2 are the top and
% bottom pixel coordinates of an autorickshaw in img/000001.jpg. We
% compute the cooresponding XYZ points in units (not meters because no
% depth info is given). Then we can use the factual information that auto
% is 1.7m high to map P1-P2 to 1.7m

%% Step 1: Invert KRt. Store as a matrix.

% Intrinsic Matrix
K =  [802.6019072886633, 0.0, 695.9898828782235; 0.0, 802.022645475923, 403.51068922309906; 0.0, 0.0, 1.0];
%  Projection Martix
P = [614.978699, 0.000000, 697.573780, 0.000000; 0.000000, 685.052002, 378.783433, 0.000000; 0.000000, 0.000000, 1.000000, 0.000000];
inverse_proj = pinv(K*P);

%% Find key points manually and store as an array
img = imread('img/000001.jpg');

p1 = [709;405;1];
p2 = [709;565;1];
pixel_dist = abs(p2(2) - p1(2));

%% Apply inverse projection.
P1 = inverse_proj*p1;
P2 = inverse_proj*p2;
XYZ_dist = abs(P1(2) - P2(2));

%% Mapping
vertical_mapping = XYZ_dist/pixel_dist;

