function status = svm_classify(options, data, model, predictions)
% SVM_CLASSIFY - Interface to SVM light, classification module
%   
%   STATUS = SVM_CLASSIFY(OPTIONS, DATA, MODEL, PREDICTIONS)
%   Call the classification program 'svm_classify' of the SVM light
%   package.
%   OPTIONS must be a structure generated by SVMLOPT. For SVM_CLASSIFY,
%   only the options 'ExecPath' and 'Verbosity' are relevant.
%   DATA is the name of the file containing the data to be classified
%   (the test data). Use SVMLWRITE to convert a Matlab matrix to the
%   appropriate format.
%   MODEL is the name of the file holding the trained Support Vector
%   Machine, generated by SVM_LEARN.
%   PREDICTIONS is the file that will store the predicted classes. There
%   is one line per test example in output_file containing the value of
%   the decision function on that example. The sign of this value
%   determines the predicted class. This file can be read into Matlab
%   using SVMLREAD.
%   If 'svm_learn' is not on the path, OPTIONS must contain a field
%   'ExecPath' with the path of the executable.
%   STATUS is the error code returned by SVM light (0 if everything went
%   fine)
%
%   See also SVML, SVMLOPT, SVMLWRITE, SVMLREAD, SVM_LEARN
%

% 
% Copyright (c) by Anton Schwaighofer (2001)
% $Revision: 1.6 $ $Date: 2002/08/09 20:24:12 $
% mailto:anton.schwaighofer@gmx.net
% 
% This program is released unter the GNU General Public License.
% 

error(nargchk(4, 4, nargin));

Names = fieldnames(options);
[m,n] = size(Names);

s = '';
for i = 1:m,
  field = Names{i,:};
  value = getfield(options, field);
  switch field,
    case 'Verbosity'
      s = stroption(s, '-v %i', value);
  end
end

evalstr = [fullfile(options.ExecPath, 'svm_classify') s ' ' ...
           data ' ' model ' ' predictions];
% fprintf('\nCalling SVMlight:\n%s\n\n', evalstr);
if isunix,
  status = unix(evalstr);
else
  status = dos(evalstr);
end


function s = stroption(s, formatstr, value, varargin)
% STROPTION - Add a new option to string
% 

if ~isempty(value),
  s = [s ' ' sprintf(formatstr, value, varargin{:})];
end
