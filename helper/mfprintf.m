function mfprintf(fileId, varargin)
%mfprintf fprintf to console and text-file
% 
% @author Aline Sindel
%
fprintf(varargin{:});
fprintf(fileId, varargin{:});
end