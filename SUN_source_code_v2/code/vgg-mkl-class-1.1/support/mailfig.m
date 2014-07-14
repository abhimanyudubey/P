function mailfig(hs, subject)
% MAILFIG  Mail figures
%   MAILFIG uses SENDMAIL.PY and the configuration file .MAILFIG in
%   the current directory to send by mail the current figure.
%
%   MAILFIG(HS) specifies a list of figures to send.
%
%   The configuration file MAILFIG contains options for
%   SENDMAIL.PY, such as:
%
%     --from      FROM WHOM
%     --to        TO WHOM
%     --server    SMTP SERVER
%     --username  SMTP TSL USERNAME
%     --password  SMTP TSL PASSWORD
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

cmd = fullfile(fileparts(mfilename('fullpath')), 'sendmail.py') ;

if nargin < 1
  hs = gcf ;
end

if nargin < 2
    mailsubject = 'Mailfig';
else
    mailsubject = sprintf('Mailfig: %s', subject);
end

files = '' ;
files_= {} ;
figs='' ;

[haveEpstopdf, r] = unix('which epstopdf') ;
haveEpstopdf = ~ haveEpstopdf ;

for h=hs
  tmp_pdf=sprintf('/tmp/figure-%d.pdf',h) ;
  if haveEpstopdf
    tmp_eps=sprintf('/tmp/figure-%d.eps',h) ;
    print(h,'-depsc',tmp_eps) ;
    [s,r]=unix(sprintf('LD_LIBRARY_PATH= epstopdf ''%s'' --outfile ''%s''', tmp_eps, tmp_pdf)) ;
    delete(tmp_eps) ;
    fprintf(r) ;
    if s
      fprintf(r) ;
      warning('EPSTOPDF failed! Printing PDF directly.') ;
      print(h,'-dpdf',tmp_pdf) ;
    end
  else
    print(h,'-dpdf',tmp_pdf) ;
  end
  files_ {end+1} = tmp_pdf ;
  files = [files sprintf(' ''%s''', tmp_pdf)] ;
end

unix(sprintf(...
  'LD_LIBRARY_PATH= python %s `cat .mailfig` --subject ''%s'' %s', ...
  cmd, mailsubject, files)) ;

delete(files_{:}) ;
