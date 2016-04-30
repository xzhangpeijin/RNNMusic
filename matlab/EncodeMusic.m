% Script for encoding music files into feature representation
clear, clc, close all

% Arguments
window = 0.04;
hop = 0.01;
bins = 256;

music_dir = '../data/music';
output_dir = '../data/encoded/';

for file = get_files(music_dir, '.flac')
    [~, name, ~] = fileparts(file{:});
    encode(file{:}, [output_dir name '.csv'], window, hop, bins);
end
