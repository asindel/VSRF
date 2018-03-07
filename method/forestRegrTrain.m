function forest = forestRegrTrain( Xfeat, Xsrc, Xtar, opts )
% forestRegrTrain Trains a random regression forest. 
% 
% The function trains a random regression forest that learns mappings from
% the low-resolution (LR) feature vectors (Xfeat) to the high-resolution 
% (HR) feature vectors (Xtar). 
%
% Based on [1], there is also the possiblity to choose the feature vectors
% only for seperation of the data into subsets while at the leaves the
% low-resolution patches (Xsrc) are utilized to learn the mapping function.
%
% To use the same LR feature vectors (Xfeat) for both learning the tree
% structure and learning the mapping in the leaves, the function can be
% called with Xsrc=[]; 
% forestRegrTrain( Xfeat, [], Xtar, opts ); 
%
% Dimensions:
%  M    - number trees
%  N    - number input vectors
%  Ff   - dimensionality of Xfeat
%  Fs   - dimensionality of Xsrc
%  Ft   - dimensionality of Xtar
%
% USAGE
%  opts   = forestRegrTrain( )
%  forest = forestRegrTrain( Xfeat, [], Xtar, opts )
%  forest = forestRegrTrain( Xfeat, Xsrc, Xtar, opts )
%
% INPUTS
%  Xfeat      - [REQ: Ff x N] N length Ff feature vectors
%  Xsrc       - [REQ: Fs x N] N length Fs source vectors
%  Xtar       - [REQ: Ft x N] N length Ft target vectors
%  opts       - additional params (struct)
%   .M        - [30] number of trees to train
%   .minCount - [128] minimum number of data points to allow split
%   .minChild - [64] minimum number of data points allowed at child nodes
%   .N1       - [N*.75] number of data points for training each tree
%   .F1       - [sqrt(F)] number features to sample for each node split
%   .F2       - [5] number of thresholds to sample for each node split. If
%               F2=0, the median of the responses is chosen. 
%   .maxDepth - [64] maximum depth of tree
%   .fWts     - [] weights used for sampling features
%   .splitfuntype - ['pair'] split function type: single or pair tests
%   .nodesubsample - [512] data subsampling on the node level. 0 means no
%               subsampling is done. Values > 0 indicate the size of the 
%               subsample
%   .splitevaltype - ['variance'] Type of split function evaluation:
%               three options are available: 'balanced', 'variance',
%               'reconstruction'
%   .lambda   - [.01] regularization parameter least squares problems
%               (splitting and leafs)
%   .estimatelambda [1] - try to estimate lambda automatically
%   .kappa     - [1] regularization parameter for split quality
%   .leaflearntype - ['linear'] dictionary learning variant for the leaf
%                    nodes: constant, linear, polynomial
%   .usepf    - [0] use parfor for training trees
%   .verbose  - [0] verbosity level (only 0 and 1 available)
%
% OUTPUTS
%  forest     - learned forest model struct array with the following fields
%   .fids         - [Kx(1 or 2)] feature ids for each node
%   .thrs         - [Kx1] threshold corresponding to each fid
%   .child        - [Kx1] index of child for each node
%   .count        - [Kx1] number of data points at each node
%   .depth        - [Kx1] depth of each node
%   .leafids      - [Kx1] leaf ids
%   .leafmapping  - [N_leafx1] N_leaf regression matrices of dimension FtxFs
%
% See also: forestRegrApply, 
% 
% The code is adapted from [1] and Piotr's Image&Video Toolbox (Copyright 
% Piotr Dollar). 
%
% @author Aline Sindel
%
% REFERENCES
% [1] S. Schulter, C. Leistner, and H. Bischof. Fast and accurate image 
% upscaling with super-resolution forests. CVPR 2015.

% get additional parameters and fill in remaining parameters
if nargin < 4, opts = struct; end
if ~isfield(opts, 'M'),opts.M = 30; end
if ~isfield(opts, 'minChild'),opts.minChild = 64; end
if ~isfield(opts, 'minCount'),opts.minCount = 128; end
if ~isfield(opts, 'N1'),opts.N1 = []; end
if ~isfield(opts, 'F1'),opts.F1 = []; end
if ~isfield(opts, 'F2'),opts.F2 = 5; end
if ~isfield(opts, 'maxDepth'),opts.maxDepth = 15; end
if ~isfield(opts, 'fWts'),opts.fWts = []; end
if ~isfield(opts, 'splitfuntype'),opts.splitfuntype = 'pair'; end
if ~isfield(opts, 'nodesubsample'),opts.nodesubsample = 512; end
if ~isfield(opts, 'splitevaltype'),opts.splitevaltype = 'variance'; end
if ~isfield(opts, 'lambda'),opts.lambda = 0.01; end
if ~isfield(opts, 'estimatelambda'),opts.estimatelambda = 1; end
if ~isfield(opts, 'kappa'),opts.kappa = 1; end
if ~isfield(opts, 'leaflearntype'),opts.leaflearntype = 'linear'; end
if ~isfield(opts, 'usepf'),opts.usepf = 0; end
if ~isfield(opts, 'verbose'),opts.verbose = 0; end
if nargin == 0, forest=opts; return; end

[Ff,N]=size(Xfeat); 
if ~isempty(Xsrc), [~,Ncheck]=size(Xsrc); assert(N==Ncheck); end
[~,Ncheck]=size(Xtar); assert(N==Ncheck);
opts.minChild=max(1,opts.minChild); opts.minCount=max([1 opts.minCount opts.minChild]);
if(isempty(opts.N1)), opts.N1=round(N*.75); end; opts.N1=min(N,opts.N1);
if(isempty(opts.F1)), opts.F1=round(sqrt(Ff)); end; opts.F1=min(Ff,opts.F1);
if(opts.F2<0), error('F2 should be > -1'); end
if(isempty(opts.fWts)), opts.fWts=ones(1,Ff,'single'); end; opts.fWts=opts.fWts/sum(opts.fWts);
if(opts.nodesubsample<2*opts.minChild), error('nodesubsample < 2*minChild'); end
if(opts.nodesubsample<3*opts.minChild), warning('nodesubsample < 3*minChild'); end

% make sure data has correct types
if(~isa(Xfeat,'single')), Xfeat=single(Xfeat); end
if ~isempty(Xsrc) && (~isa(Xsrc,'single')), Xsrc=single(Xsrc); end
if(~isa(Xtar,'single')), Xtar=single(Xtar); end
if(~isa(opts.fWts,'single')), opts.fWts=single(opts.fWts); end

% train M random trees on different subsets of data
dWtsUni = ones(1,N,'single'); dWtsUni=dWtsUni/sum(dWtsUni);
tree = struct('fids',[],'thrs',[],'child',[],'count',[],'depth',[],...
  'leafids',[],'leafmapping',[]);
forest = tree(ones(opts.M,1));  

if opts.usepf
  %numCores = feature('numcores');  
  if isfield(opts, 'Npworkers') 
    parpool(opts.Npworkers); %specify number of matlab parallel workers
  end
  parfor i=1:opts.M
      if N==opts.N1 %#ok<PFBNS> %use all samples
          forest(i) = treeRegrTrain(Xfeat,Xsrc,Xtar,opts);
      else %subsample
          d=wswor(dWtsUni,opts.N1,4); 
          Xfeat1=Xfeat(:,d); 
          if ~isempty(Xsrc)
              Xsrc1=Xsrc(:,d); 
          else
              Xsrc1=[];
          end
          Xtar1=Xtar(:,d);
          forest(i) = treeRegrTrain(Xfeat1,Xsrc1,Xtar1,opts); 
      end
  end  
else
  for i=1:opts.M
    if N==opts.N1
        forest(i) = treeRegrTrain(Xfeat,Xsrc,Xtar,opts);
    else
        d=wswor(dWtsUni,opts.N1,4); %subsample
        if ~isempty(Xsrc)
            forest(i) = treeRegrTrain(Xfeat(:,d),Xsrc(:,d),Xtar(:,d),opts);
        else
            forest(i) = treeRegrTrain(Xfeat(:,d),[],Xtar(:,d),opts);
        end
    end
  end
  memUsed
end
end


% =========================================================================
% ========= helper function ===============================================

function tree = treeRegrTrain( Xfeat, Xsrc, Xtar, opts )
% Train a single regression tree

% define some constants and the tree model
[~,N]=size(Xfeat); K=2*N-1;
thrs=zeros(K,1,'single'); 
if strcmp(opts.splitfuntype,'single')
  fids=zeros(K,1,'uint32');
elseif strcmp(opts.splitfuntype,'pair')
  fids=zeros(K,2,'uint32');
else
  error('Unknown splitfunction type');
end
child=zeros(K,1,'uint32'); count=child; depth=child;
leafmapping = cell(K,1);
dids=cell(K,1); dids{1}=uint32(1:N); k=1; K=2;
% frequence for printing a status message (if in verbose mode)
msgnodestep = 200;
useOnlyXfeat = isempty(Xsrc);

% train the tree
while( k < K )
  % get node data
  dids1=dids{k}; count(k)=length(dids1); 
  XfeatNode=Xfeat(:,dids1); XtarNode = Xtar(:,dids1);
  if ~useOnlyXfeat, XsrcNode = Xsrc(:,dids1); end
  if (opts.verbose && (mod(k,msgnodestep)==0||k==1)), ...
      fprintf('Node %04d, depth %02d, %07d samples () ',k,depth(k),count(k)); end
  
  % if insufficient data or max-depth reached, don't train split
  if( count(k)<=opts.minCount||depth(k)>opts.maxDepth||count(k)<(2*opts.minChild) )
    if (opts.verbose && (mod(k,msgnodestep)==0)||k==1), ...
        fprintf('becomes a leaf (stop criterion active)\n'); end
    if useOnlyXfeat
        leafmapping{k}=createLeaf(XfeatNode,XtarNode,opts.leaflearntype,opts.lambda,opts.estimatelambda);
    else
        leafmapping{k}=createLeaf(XsrcNode,XtarNode,opts.leaflearntype,opts.lambda,opts.estimatelambda);
    end
    k=k+1; continue;
  end
  if (opts.verbose && (mod(k,msgnodestep)==0||k==1)), fprintf('find split () '); end
  
  % compute responses for all data samples
  switch opts.splitfuntype
    case 'single'
      fids1=wswor(opts.fWts,opts.F1,4); resp=XfeatNode(fids1,:);
    case 'pair'
      fids1 = [wswor(opts.fWts,opts.F1,4); wswor(opts.fWts,opts.F1,4)];
      % Caution: same feature id could be sampled  -> all zero responses
      resp=XfeatNode(fids1(1,:),:)-XfeatNode(fids1(2,:),:);
    otherwise
      error('Unknown splitfunction type');
  end
  
  % subsample the data for splitfunction node optimization
  if opts.nodesubsample > 0 && opts.nodesubsample < count(k)
    randinds = randsample(count(k),opts.nodesubsample);
    respSub = resp(:,randinds); 
    if useOnlyXfeat
        XsrcSub = XfeatNode(:,randinds); 
    else
        XsrcSub = XsrcNode(:,randinds); 
    end
    XtarSub = XtarNode(:,randinds);
  else
    respSub = resp; 
    if useOnlyXfeat, XsrcSub = XfeatNode; else, XsrcSub = XsrcNode; end
    XtarSub = XtarNode;
  end
  if (opts.verbose && (mod(k,msgnodestep)==0||k==1)), ...
      fprintf('subsmpl = %07d/%07d () ',size(respSub,2),size(resp,2)); end
  
  % find best splitting function and corresponding threshold
  [fid,thr,rerr]=findSplitAndThresh(respSub,XsrcSub,XtarSub,opts.F2,...
    opts.splitevaltype,opts.lambda,opts.minChild,opts.kappa);
  
  % check validity of the splitting function
  validsplit=false;
  left=resp(fid,:)<thr; count0=nnz(left); fid=fids1(:,fid);
  if ~isinf(rerr) && count0>=opts.minChild && (count(k)-count0)>=opts.minChild
    validsplit=true;
  end
  
  % continue tree training (either split or create a leaf)
  if validsplit
    child(k)=K; fids(k,:)=fid; thrs(k)=thr; dids{K}=dids1(left); 
    dids{K+1}=dids1(~left); depth(K:K+1)=depth(k)+1; K=K+2; 
    dids{k}=[]; % delete the dids as we have a split node here
    if (opts.verbose && (mod(k,msgnodestep)==0||k==1)), ...
        fprintf('valid split (loss=%.6f)\n',rerr); end
  else
    if (opts.verbose && (mod(k,msgnodestep)==0||k==1)), ...
        fprintf('invalid split -> leaf\n'); end
    if useOnlyXfeat
        leafmapping{k}=createLeaf(XfeatNode,XtarNode,opts.leaflearntype,opts.lambda,opts.estimatelambda);
    else
        leafmapping{k}=createLeaf(XsrcNode,XtarNode,opts.leaflearntype,opts.lambda,opts.estimatelambda);
    end
  end
  k=k+1;
end
K=K-1;

% create the leaf-node id mapping
leafids = ones(K,1,'int32')*(-1);
leafcnt = 0;
for i=1:K
  if ~isempty(leafmapping{i})       
      leafids(i) = leafcnt; 
      leafcnt = leafcnt + 1; 
  end
end
%reduce leaf mapping field to non-empty cells
myleafmapping = leafmapping(~cellfun('isempty',leafmapping));
leafmapping = cell(1);
leafmapping{1} = myleafmapping;
fids = fids - 1;

% create output model struct
tree=struct('fids',fids(1:K,:),'thrs',thrs(1:K),'child',child(1:K),...
  'count',count(1:K),'depth',depth(1:K),'leafids',leafids(1:K),'leafmapping',leafmapping);


end

function [fid, thr, rerr] = findSplitAndThresh( resp, Xsrc, Xtar, F2, splitevaltype, lambda, minChild, kappa )
[F1,~]=size(resp); rerr=Inf; fid=1; thr=Inf; Ft=size(Xtar,1); Fs=size(Xsrc,1);
% special treatment for random tree growing
if strcmp(splitevaltype,'random'), F1=1; F2=1; end
% iterate the random split functions
for s=1:F1
  % get thresholds to evaluate
  if F2==0, tthrs=median(resp(s,:));
  else
    respmin=min(resp(s,:)); respmax=max(resp(s,:));
    tthrs = zeros(F2+1,1,'single'); % we also add the median as threshold
    tthrs(1:end-1) = rand(F2,1)*0.95*(respmax-respmin) + respmin;
    tthrs(end) = median(resp(s,:));
  end
  % iterate the thresholds
  for t=1:length(tthrs)
    tthr=tthrs(t); left=resp(s,:)<tthr; right=~left; 
    nl=nnz(left); nr=nnz(right);
    if nl<minChild || nr<minChild, continue; end
    XsrcL=Xsrc(:,left); XsrcR=Xsrc(:,right);
    XtarL=Xtar(:,left); XtarR=Xtar(:,right);
    % compute the quality if the splitting function
    switch splitevaltype
      case 'random'
        trerr = 0; % this is better than Inf (it can be constant because we only evaluate once)
      case 'balanced'
        trerr = (nl - nr)^2;
      case 'variance'
        trerrL = sum(var(XtarL,0,2))/Ft; 
        trerrR = sum(var(XtarR,0,2))/Ft;
        if kappa>0
          trerrLsrc=sum(var(XsrcL,0,2))/Fs; 
          trerrRsrc=sum(var(XsrcR,0,2))/Fs;
          trerrL=(trerrL+kappa*trerrLsrc)/2; 
          trerrR=(trerrR+kappa*trerrRsrc)/2;
        end
        trerr = (nl*trerrL + nr*trerrR)/(nl+nr);
      case 'reconstruction' % based on a sampled dictionary
        XsrcL = [XsrcL; ones(1,size(XsrcL,2),'single')]; %#ok<AGROW>
        TL =  XtarL * ((XsrcL*XsrcL' + lambda*eye(size(XsrcL,1))) \ XsrcL)';
        XsrcR = [XsrcR; ones(1,size(XsrcR,2),'single')]; %#ok<AGROW>
        TR =  XtarR * ((XsrcR*XsrcR' + lambda*eye(size(XsrcR,1))) \ XsrcR)';
        trerrL = sqrt(sum(sum((XtarL-TL*XsrcL).^2))/nl);
        trerrR = sqrt(sum(sum((XtarR-TR*XsrcR).^2))/nr);
        if kappa > 0% regularizer
          trerrLsrc=sum(var(XsrcL,0,2))/Fs; trerrRsrc=sum(var(XsrcR,0,2))/Fs;
          trerrL=(trerrL+kappa*trerrLsrc)/2; trerrR=(trerrR+kappa*trerrRsrc)/2;
        end
        trerr = (nl*trerrL + nr*trerrR) / (nl+nr);
      otherwise
        error('Unknown split evaluation type');
    end

    if trerr<rerr, rerr=trerr; thr=tthr; fid=s; end
  end
end

end

function T = createLeaf( Xsrc, Xtar, leaflearntype, lambda, autolambda )
% creates a leaf node and computes the prediction model
switch leaflearntype
  case 'constant'
    T = sum(Xtar,2)/size(Xtar,2); %predmodeltype = 0;
  case 'linear'
    warning('off','all'); %suppress warning: Matrix is close to singular or badly scaled. Results may be inaccurate.
    Xsrc = [Xsrc; ones(1,size(Xsrc,2),'single')];
    matinv = Xsrc*Xsrc'; if autolambda, lambda=estimateLambda(matinv); end    
    T = Xtar * ((matinv + lambda*eye(size(Xsrc,1))) \ Xsrc)';
%     %%% Check if mapping function is NaN 
%     nanT = isnan(T);
%     if sum(nanT(:))>0
%         fprintf('Mapping function is NaN');
%     end
  case 'polynomial' % actually, it is only polynomial with quadratic term :)
    Xsrc = [Xsrc; ones(1,size(Xsrc,2),'single'); Xsrc.^2];
    matinv = Xsrc*Xsrc'; if autolambda, lambda=estimateLambda(matinv); end
    T = Xtar * ((matinv + lambda*eye(size(Xsrc,1))) \ Xsrc)';
  otherwise
    error('Unknown leaf node prediction type');
end
end

function lambda = estimateLambda(matinv)
% Code from Schulter et al. [1]
rcondTmpMat = rcond(matinv); if rcondTmpMat<eps, rcondTmpMat = 1e-10; end
lambda = 0;
if rcondTmpMat < 1e-2, lambda = 10^(-4 - log10(rcondTmpMat)) * rcondTmpMat; end
if isnan(lambda), error('This case should never occur!'); end
end

function ids = wswor( prob, N, trials )
% Fast weighted sample without replacement. Alternative to:
%  ids=datasample(1:length(prob),N,'weights',prob,'replace',false);
% Code from Schulter et al. [1]
M=length(prob); 
assert(N<=M); 
if(N==M), ids=1:N; return; end
if(all(prob(1)==prob))
    ids=randperm(M); 
    ids=ids(1:N); 
    return; 
end
cumprob=min([0 cumsum(prob)],1); 
assert(abs(cumprob(end)-1)<.01);
cumprob(end)=1; 
[~,ids]=histc(rand(N*trials,1),cumprob);
[s,ord]=sort(ids); 
K(ord)=[1; diff(s)]~=0; 
ids=ids(K);
if(length(ids)<N)
    ids=wswor(cumprob,N,trials*2); 
end
ids=ids(1:N)';
end
