function [] = encode(audio, output, window, hop, bins, limit)
%ENCODE Encodes a music file into feature representation
%   input - input audio file
%   output - output text file
%   window - window length of fft transform (in seconds)
%   overlap - overlap between windows
%   bins - number of frequency bins
%   limit (optional) - limit time to encode (in seconds)
%   Default settings: window = 0.04, hop = 0.01, bins = 256
    [y, Fs] = audioread(audio);
    if size(y,2) == 2
        % If y is dual channel, average to one channel
        y = (y(:,1) + y(:,2)) / 2;
    end
    
    % Normalize
    y = y / max(abs(y));
        
    % We should be reading in 44.1kHz music
    assert(Fs == 44100);
    
    % Downsample to 22.05 kHz since we're looking at at most 8kHz freq
    y = downsample(y, 2);
    Fs = Fs / 2;
    
    % Truncate music
    if exist('limit', 'var')
       y = y(1:min(length(y), round(Fs*limit)));
    end

    % Perform fft
    eps = 1e-7;
    [s, f, t] = spectrogram(y, round(window * Fs - eps), ...
        round((window - hop) * Fs - eps), 2*bins - 1, Fs);
    
    % Convert to polar form
    [theta, rho] = cart2pol(real(s), imag(s));
    
    % Convert to db
    rho = 10 + log10(rho .^2);
    
    % Format and write to csv
    M = vertcat([Fs transpose(f) transpose(f)], ...
        horzcat(transpose(t), transpose(rho), transpose(theta)));
    csvwrite(output, M);
end

