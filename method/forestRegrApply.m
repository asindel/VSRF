function XtarPred = forestRegrApply( Xfeat, Xsrc, forest, leaflearntype, Mhat, NCores )
% forestRegrApply Applies a trained regression forest for volumetric 
% super-resolution of MRI data. 
% 
% The function applies the trained random forest to predict the target 
% vectors for the low-resolution feature vectors (Xfeat).
% XSrc are either a copy of low-resolution feature vectors (Xfeat) or the
% low-resolution patches.
% 
% The parameter Mhat defines the number of trees (out of the model) that
% are really evaluated. NCores defines the number of CPU cores that are
% used for parallelizing the inference procedure. 
%
% INPUTS
%  Xfeat            - [FfxN] N length Ff feature vectors
%  Xsrc             - [FsxN] N length Fs data vectors (or the same as Xfeat)
%  forest           - learned regression forest model
%  leaflearntype    - type for leaf prediction model
%  Mhat             - [length(forest)] number of trees used for inference
%  NCores           - #CPU cores that should be used for inference
%
% OUTPUTS
%  XtarPred - [FtxN] N length Ft predicted target vectors
%
% See also forestRegrTrain
%
% The code is adapted from [1] and Piotr's Image&Video Toolbox (Copyright 
% Piotr Dollar). 
%
% @author Aline Sindel
%
% REFERENCES
% [1] S. Schulter, C. Leistner, and H. Bischof. Fast and accurate image 
% upscaling with super-resolution forests. CVPR 2015.

assert(isa(Xfeat,'single')); M=length(forest); nthreads = 1;
if nargin >= 4
  if (Mhat < 1 || Mhat > M), error('Mhat is set wrong: 0 < Mhat <= M'); end
else
  Mhat = length(forest);
end
if nargin >= 5, nthreads = NCores; if NCores<1, error('NCores set below 1!'); end; end
assert(~isempty(forest));

[~,Fs]=size(forest(1).leafmapping{1,1});

if strcmp(leaflearntype,"constant")
    leafpredtype = 0;
elseif strcmp(leaflearntype,"linear")
    leafpredtype = 1;
elseif strcmp(leaflearntype,"polynomial")
    leafpredtype = 2;
end

switch leafpredtype
  case 0
    assert(Fs==1);
  case 1
    assert(Fs==size(Xsrc,1)+1); % +1 for bias term!
    Xsrc = [Xsrc; ones(1,size(Xsrc,2),'single')];
  case 2
    assert(Fs==size(Xsrc,1)*2+1); % *2 + 1 for doubled size and bias term
    Xsrc = [Xsrc; ones(1,size(Xsrc,2),'single'); Xsrc.^2];
  otherwise
    error('Unknown leaf node prediction model');
end

% iterate the trees
myforest = forest(1:Mhat); 

XtarPred = forestRegrInference(Xfeat',Xsrc',myforest,leafpredtype,nthreads);
end