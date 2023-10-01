% setpaths.m
%
% Add necessary paths

if ~exist('multisoftplus','file')
    addpath utils
end

if ~exist('neglogli_multinomGLM_reduced','file')
    addpath inference
end
