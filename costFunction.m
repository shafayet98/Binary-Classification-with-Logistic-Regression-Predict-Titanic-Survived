function [cost_J, grad] = costFunction(theta,X,y)

    m = length(X);
    g = sigmoid(X*theta);

    cost_J = (-1/m) * sum( (y .* log(g)) + ( (1-y) .* log(1-g) ) );

    grad = (1/m) * (X' * (g-y));
    

end;