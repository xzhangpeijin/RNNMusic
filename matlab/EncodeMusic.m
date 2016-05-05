% Script for encoding music files into feature representation
clear, clc, close all

% Arguments
window = 0.04;
hop = 0.01;
bins = 256;

albums = {'[1969] Abbey Road (FLAC)', ...
    'Daft Punk - 2013 - Random Access Memories (Vinyl FLAC 16-44)', ...
    'Steve Reich - Third Coast Percussion [2016] Flac', ...
    'The Classical Collection 3 Chopin - Piano Classics'};
           
for album = albums
    music_dir = ['../data/music/' album{:}];
    output_dir = ['../data/encoded/' album{:}];

    for file = get_files(music_dir, '.flac')
        [~, name, ~] = fileparts(file{:});
        encode(file{:}, [output_dir '/' name '.csv'], window, hop, bins, 30);
    end
end
