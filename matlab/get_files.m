function [files] = get_files(path, extn)
    files = {};
    A = dir(path);
    for i = 1:length(A)       
       [~, name, ext] = fileparts(A(i).name);
       if strcmp(ext, extn) == 1
           files{end + 1} = [path '/' A(i).name];
       elseif A(i).name(1) ~= '.' && A(i).isdir
           files = [files get_files([path '/' A(i).name], extn)];
       end
    end
end

