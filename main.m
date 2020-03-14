% load data
data_train = csvread("train_data.csv");
data_test = csvread("test_data.csv");

% extracting data from csv and creating training and test variable
X_train = data_train(2:length(data_train),4:17);
y_train = data_train(2:length(data_train),3);

X_test = data_test(2:length(data_test), 4:17);
y_test = data_test(2:length(data_test), 3);

% adding x0 feature as 1
[m n] = size(X_train);
[m_test n_test] = size(X_test);

X_train = [ones(m,1) X_train];
X_test = [ones(m_test,1) X_test];

% initializing theta
init_theta = zeros(n+1,1);

% displaying the size of X, y and init_theta
disp(size(X_train));
disp(size(y_train));
disp(size(init_theta));

% compute cost and gradiant
[cost_J, grad] = costFunction(init_theta,X_train,y_train);

% using function minimization unconstrain (fminunc) to minimize theta 
options = optimset('GradObj','on','MaxIter',400);
[theta, cost] = fminunc(@(t)(costFunction(t, X_train, y_train)), init_theta, options); % return theta and cost

% testing
prob = sigmoid(X_test*theta);
probability_modify = probMod(prob);
probability_modify = probability_modify';

compare = [probability_modify y_test];
fprintf("Showing the first 10 comparison (predicted vs already labeled) \n")
disp(compare(1:10,:));


% Training Accuracy
fprintf('Train Accuracy: %f\n', mean((probability_modify == y_test)) * 100);

