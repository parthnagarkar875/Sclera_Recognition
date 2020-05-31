function wait2()
h = waitbar(0,'Neighbourhood Matching...');
steps = 1200;
for step = 1:steps
    % computations take place here
    waitbar(step / steps)
end
close(h) 


end

