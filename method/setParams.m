function opts = setParams(params)
% setParams creates a struct with the parameters for super-resolution
% random forests with 3-D patches
% Input:
% params: struct with parameter options (otherwise set by default)
%
% @author Aline Sindel
%

opts = struct;

if nargin > 0
    opts = setOptions(opts, params);
end

%set remaining parameters
if ~isfield(opts, 'sf'),opts.sf = 2; end
if ~isfield(opts, 'scaleDim'), opts.scaleDim = 3; end
if ~isfield(opts, 'interpolMethod'),opts.interpolMethod = 'tricubic'; end
if ~isfield(opts, 'downsamplingMethod'),opts.downsamplingMethod = 'tricubic'; end
if ~isfield(opts, 'ensemble'),opts.ensemble = 'median'; end

if ~isfield(opts, 'verbose'),opts.verbose = 1; end

if ~isfield(opts, 'pRegrForest')
    opts.pRegrForest = forestRegrTrain(); 
end
      
if ~isfield(opts, 'patchSize'), opts.patchSize = [1 1 1] * opts.sf + [1 1 1]; end
if ~isfield(opts, 'patchStride'), opts.patchStride = opts.patchSize - [1 1 1]; end
if ~isfield(opts, 'patchBorder'), opts.patchBorder = [0 0 0]; end
% Features
if ~isfield(opts, 'patchfeats')
    opts.patchfeats.type = 'filtersDevEdge';
end
if ~strcmp(opts.patchfeats.type, 'none') && ~isfield(opts.patchfeats, 'filters')
    O = zeros(1, opts.sf-1);
    G = [1 O -1]; % Gradient
    D = [1 O -2 O 1]/2; % 2nd derivative        
    G3 = reshape(G, [1,1,size(G,2)]);
    D3 = reshape(D, [1,1,size(D,2)]);
    opts.patchfeats.filters = {G, G.', G3, D, D.', D3}; % 3D versions    
end  

end

function opts = setOptions(opts, params)
%copy parameters from params to opts
fields = fieldnames(params);
for i = 1:numel(fields)
  opts.(fields{i}) = params.(fields{i});
end
end
