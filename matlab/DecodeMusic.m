% Converts feature representation of audio files into actual audio
clear, clc, close all

output_dir = '../data/encoded';
synth_dir = '../data/synth/';

for file = get_files(output_dir, '.csv')
    [~, name, ~] = fileparts(file{:});
    decode(file{:}, [synth_dir name '.flac']);
end
