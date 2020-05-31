function wait()
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
h = waitbar(0,'Creating Database Please Wait...');
steps = 1000;
for step = 1:steps
    % computations take place here
    waitbar(step / steps)
end
close(h) 

end

