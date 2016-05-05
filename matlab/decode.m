function [] = decode(input, output)
%DECODE Decodes a feature representation of audio into an audio file
%   input - Feature file of frequency amplitudes
%   output - Output file to write audio to
    M = csvread(input);
    
    nfft = (size(M,2) - 1)/2;

    Fs = M(1,1);
    t = M(2:end,1);
    
    % Pull polar data
    rho = M(2:end,2:nfft+1);
    theta = M(2:end,nfft+2:end);
    
    % Wrap angles
    theta = wrapTo2Pi(theta) - pi;
    
    % Decode from psd
    rho = sqrt(10 .^ (rho - 10));
    
    % Convert to sfft
    [re, im] = pol2cart(theta, rho);
    s = re + 1j * im;
    
    % Reverse stft transform
    [x, ~] = istft(transpose(s), ...
        round((t(end) - t(end-1)) * Fs), 2*nfft - 1, Fs);
    
    % Normalize
    x = x / max(abs(x));

    audiowrite(output, x, Fs);
end

