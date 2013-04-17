function compile_mex_files(str_dir, str_ext)
% Compiles all mex files in current directory
%
% Edwin V. Bonilla
d = dir([str_dir, '/*.', str_ext]);
for i = 1 : length(d)
    str_fname = [str_dir, '/', d(i).name];
    str_cmd = ['mex -outdir ', str_dir, ' ', str_fname,   ';'];
    disp(str_cmd);
    eval(str_cmd);
end

return;

